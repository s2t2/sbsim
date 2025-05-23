"""Models an Air Handler Unit (AHU) within an HVAC system.

This module defines the `AirHandler` class, which simulates the behavior of a
central air handling unit. The AHU is responsible for conditioning (heating or
cooling) a mixture of recirculated indoor air and fresh outdoor air, and then
distributing this supply air to various zones in the building.
"""

from typing import Optional
import uuid

import gin

from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import smart_device
from smart_control.simulator import weather_controller as weather_controller_py
from smart_control.utils import constants


@gin.configurable
class AirHandler(smart_device.SmartDevice):
  """Simulates an Air Handler Unit (AHU) with configurable properties.

  The AHU model includes:
  - Mixing of recirculated indoor air and fresh outdoor air.
  - Heating or cooling of the mixed air to meet supply air temperature setpoints.
  - Calculation of energy consumption for fans and thermal conditioning.

  The AHU's operation is typically driven by demand from downstream components
  like VAV (Variable Air Volume) units.

  Attributes:
    recirculation (float): The proportion of air that is recirculated from the
      building, versus fresh outdoor air intake (range 0.0 to 1.0).
    air_flow_rate (float): Current total air flow rate (m^3/s) being supplied
      by the AHU. This is typically an aggregation of demands from VAVs.
    heating_air_temp_setpoint (float): The target temperature (K) for supplied
      air when heating is active.
    cooling_air_temp_setpoint (float): The target temperature (K) for supplied
      air when cooling is active.
    fan_differential_pressure (float): The pressure difference (Pa) maintained
      by the supply and exhaust fans.
    fan_efficiency (float): The electrical efficiency of the fans (0.0 to 1.0).
    cooling_request_count (int): A counter for how many downstream units (e.g.,
      VAVs) are currently requesting cooling.
    max_air_flow_rate (float): The maximum possible air flow rate (m^3/s) that
      the AHU can deliver.
    _sim_weather_controller (Optional[weather_controller_py.WeatherController]):
      An optional weather controller instance to get outside air temperature.
    _observation_timestamp (pd.Timestamp): Stores the timestamp of the last
      observation request, used for weather data.
  """

  def __init__(
      self,
      recirculation: float,
      heating_air_temp_setpoint_k: float,
      cooling_air_temp_setpoint_k: float,
      fan_differential_pressure_pa: float,
      fan_efficiency_ratio: float,
      max_air_flow_rate_m3_s: float = 8.67,
      device_id: Optional[str] = None,
      sim_weather_controller: Optional[
          weather_controller_py.WeatherController
      ] = None,
  ):
    """Initializes the AirHandler instance.

    Args:
      recirculation (float): Proportion of air recirculated (0.0 to 1.0).
      heating_air_temp_setpoint_k (float): Target supply air temperature (K)
        when heating.
      cooling_air_temp_setpoint_k (float): Target supply air temperature (K)
        when cooling.
      fan_differential_pressure_pa (float): Pressure difference (Pa) across fans.
      fan_efficiency_ratio (float): Electrical efficiency of fans (0.0 to 1.0).
      max_air_flow_rate_m3_s (float): Maximum supply air flow rate (m^3/s).
      device_id (Optional[str]): A unique identifier for this AHU. If None,
        a UUID will be generated.
      sim_weather_controller (Optional[weather_controller_py.WeatherController]):
        An instance to provide outside air temperature data. Required if
        'outside_air_temperature_sensor' is an observable field.

    Raises:
      ValueError: If `cooling_air_temp_setpoint_k` is not greater than
        `heating_air_temp_setpoint_k`.
    """
    if cooling_air_temp_setpoint_k <= heating_air_temp_setpoint_k:
      raise ValueError(
          "cooling_air_temp_setpoint_k must be greater than "
          "heating_air_temp_setpoint_k to maintain a deadband."
      )

    observable_fields_info = {
        "differential_pressure_setpoint": smart_device.AttributeInfo(
            "fan_differential_pressure", float
        ),
        "supply_air_flowrate_sensor": smart_device.AttributeInfo(
            "air_flow_rate", float
        ),
        "supply_air_heating_temperature_setpoint": smart_device.AttributeInfo(
            "heating_air_temp_setpoint", float
        ),
        "supply_air_cooling_temperature_setpoint": smart_device.AttributeInfo(
            "cooling_air_temp_setpoint", float
        ),
        "supply_fan_speed_percentage_command": smart_device.AttributeInfo(
            "supply_fan_speed_percentage", float # Calculated property
        ),
        "discharge_fan_speed_percentage_command": smart_device.AttributeInfo(
            "supply_fan_speed_percentage", float # Assuming same as supply for model
        ),
        "outside_air_flowrate_sensor": smart_device.AttributeInfo(
            "ambient_flow_rate", float # Calculated property
        ),
        "cooling_request_count": smart_device.AttributeInfo(
            "cooling_request_count", int # Should be int
        ),
    }
    if sim_weather_controller:
      observable_fields_info["outside_air_temperature_sensor"] = (
          smart_device.AttributeInfo("outside_air_temperature_sensor", float)
      )

    action_fields_info = {
        "supply_air_heating_temperature_setpoint": smart_device.AttributeInfo(
            "heating_air_temp_setpoint", float
        ),
        "supply_air_cooling_temperature_setpoint": smart_device.AttributeInfo(
            "cooling_air_temp_setpoint", float
        ),
    }

    dev_id = device_id if device_id else f"air_handler_id_{uuid.uuid4()}"
    super().__init__(
        observable_fields=observable_fields_info,
        action_fields=action_fields_info,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.AHU,
        device_id=dev_id,
    )

    # Store initial values for reset
    self._init_recirculation = recirculation
    self._init_air_flow_rate_m3_s = 0.0 # Starts with no flow demand
    self._init_heating_air_temp_setpoint_k = heating_air_temp_setpoint_k
    self._init_cooling_air_temp_setpoint_k = cooling_air_temp_setpoint_k
    self._init_fan_differential_pressure_pa = fan_differential_pressure_pa
    self._init_fan_efficiency_ratio = fan_efficiency_ratio
    self._init_cooling_request_count = 0
    self._init_max_air_flow_rate_m3_s = max_air_flow_rate_m3_s
    self._sim_weather_controller = sim_weather_controller

    # Initialize current state attributes (will be set in reset)
    self._recirculation: float = 0.0
    self._air_flow_rate: float = 0.0
    self._heating_air_temp_setpoint: float = 0.0
    self._cooling_air_temp_setpoint: float = 0.0
    self._fan_differential_pressure: float = 0.0
    self._fan_efficiency: float = 0.0
    self._cooling_request_count: int = 0
    self._max_air_flow_rate: float = 0.0
    self.reset()

  def reset(self) -> None:
    """Resets the AHU state to its initial configuration."""
    self._recirculation = self._init_recirculation
    self._air_flow_rate = self._init_air_flow_rate_m3_s
    self._heating_air_temp_setpoint = self._init_heating_air_temp_setpoint_k
    self._cooling_air_temp_setpoint = self._init_cooling_air_temp_setpoint_k
    self._fan_differential_pressure = self._init_fan_differential_pressure_pa
    self._fan_efficiency = self._init_fan_efficiency_ratio
    self._cooling_request_count = self._init_cooling_request_count
    self._max_air_flow_rate = self._init_max_air_flow_rate_m3_s
    logging.debug("AirHandler '%s' reset to initial state.", self.device_id())

  @property
  def outside_air_temperature_sensor(self) -> float:
    """Current outside air temperature (K) from the weather controller."""
    if not self._sim_weather_controller:
      raise RuntimeError(
          "Weather controller not available for AirHandler "
          f"'{self.device_id()}' to get outside air temperature."
      )
    # _observation_timestamp is set by SmartDevice when observations are requested
    return self._sim_weather_controller.get_current_temp(
        self._observation_timestamp
    )

  @property
  def recirculation(self) -> float:
    """Proportion of air recirculated (0.0 to 1.0)."""
    return self._recirculation

  @recirculation.setter
  def recirculation(self, value: float) -> None:
    self._recirculation = np.clip(value, 0.0, 1.0)

  @property
  def air_flow_rate(self) -> float:
    """Current total air flow rate (m^3/s) supplied by the AHU."""
    return self._air_flow_rate

  @air_flow_rate.setter
  def air_flow_rate(self, value: float) -> None:
    # Typically set by add_demand, but allow direct set capped by max
    self._air_flow_rate = np.clip(value, 0.0, self._max_air_flow_rate)

  @property
  def cooling_air_temp_setpoint(self) -> float:
    """Target supply air temperature (K) when cooling."""
    return self._cooling_air_temp_setpoint

  @cooling_air_temp_setpoint.setter
  def cooling_air_temp_setpoint(self, value: float) -> None:
    if value <= self._heating_air_temp_setpoint:
        logging.warning("Cooling setpoint %s K is <= heating setpoint %s K.",
                        value, self._heating_air_temp_setpoint)
    self._cooling_air_temp_setpoint = value

  @property
  def heating_air_temp_setpoint(self) -> float:
    """Target supply air temperature (K) when heating."""
    return self._heating_air_temp_setpoint

  @heating_air_temp_setpoint.setter
  def heating_air_temp_setpoint(self, value: float) -> None:
    if value >= self._cooling_air_temp_setpoint:
        logging.warning("Heating setpoint %s K is >= cooling setpoint %s K.",
                        value, self._cooling_air_temp_setpoint)
    self._heating_air_temp_setpoint = value

  @property
  def fan_differential_pressure(self) -> float:
    """Pressure difference (Pa) maintained by fans."""
    return self._fan_differential_pressure

  @fan_differential_pressure.setter
  def fan_differential_pressure(self, value: float) -> None:
    self._fan_differential_pressure = value

  @property
  def fan_efficiency(self) -> float:
    """Electrical efficiency of fans (0.0 to 1.0)."""
    return self._fan_efficiency

  @fan_efficiency.setter
  def fan_efficiency(self, value: float) -> None:
    self._fan_efficiency = np.clip(value, 0.0, 1.0)

  @property
  def cooling_request_count(self) -> int:
    """Number of VAVs currently requesting cooling from this AHU."""
    return self._cooling_request_count

  @property
  def max_air_flow_rate(self) -> float:
    """Maximum possible air flow rate (m^3/s) for this AHU."""
    return self._max_air_flow_rate

  def get_mixed_air_temp(
      self, recirculation_temp_k: float, ambient_temp_k: float
  ) -> float:
    """Calculates the temperature of mixed air before conditioning.

    Args:
      recirculation_temp_k (float): Temperature (K) of the recirculated air
        from the building.
      ambient_temp_k (float): Temperature (K) of the outdoor (ambient) air.

    Returns:
      float: The temperature (K) of the air mixture.
    """
    return (
        self._recirculation * recirculation_temp_k +
        (1.0 - self._recirculation) * ambient_temp_k
    )

  def get_supply_air_temp(
      self, recirculation_temp_k: float, ambient_temp_k: float
  ) -> float:
    """Calculates the final supply air temperature after conditioning.

    The mixed air is heated or cooled to meet the respective setpoints if its
    temperature falls outside the deadband defined by
    `heating_air_temp_setpoint` and `cooling_air_temp_setpoint`.

    Args:
      recirculation_temp_k (float): Temperature (K) of recirculated air.
      ambient_temp_k (float): Temperature (K) of outdoor air.

    Returns:
      float: The final temperature (K) of the air supplied to the zones.
    """
    mixed_air_temp_k = self.get_mixed_air_temp(
        recirculation_temp_k, ambient_temp_k
    )
    if mixed_air_temp_k > self._cooling_air_temp_setpoint:
      return self._cooling_air_temp_setpoint
    elif mixed_air_temp_k < self._heating_air_temp_setpoint:
      return self._heating_air_temp_setpoint
    else: # Within deadband, no active heating/cooling of mixed air
      return mixed_air_temp_k

  @property
  def ambient_flow_rate(self) -> float:
    """Calculated fresh outdoor air flow rate (m^3/s)."""
    return (1.0 - self._recirculation) * self._air_flow_rate

  @property
  def recirculation_flow_rate(self) -> float:
    """Calculated recirculated indoor air flow rate (m^3/s)."""
    return self._recirculation * self._air_flow_rate

  @property
  def supply_fan_speed_percentage(self) -> float:
    """Calculated supply fan speed as a percentage of max flow rate."""
    if self.max_air_flow_rate == 0:
      return 0.0
    return np.clip(self._air_flow_rate / self.max_air_flow_rate, 0.0, 1.0)

  def reset_demand(self) -> None:
    """Resets accumulated air flow demand and cooling requests for a new step."""
    self._air_flow_rate = 0.0
    self._cooling_request_count = 0

  def add_demand(self, flow_rate_m3_s: float) -> None:
    """Adds to the current air flow rate demand from downstream units (VAVs).

    The total air flow rate is capped at `max_air_flow_rate`. This method
    also increments the `cooling_request_count` (though the name implies it's
    only for cooling, it's used as a general demand counter here).

    Args:
      flow_rate_m3_s (float): Air flow rate (m^3/s) to add to the demand.

    Raises:
      ValueError: If `flow_rate_m3_s` is not positive.
    """
    if flow_rate_m3_s <= 0:
      # Allow zero flow rate if a VAV is closed, but log if negative.
      if flow_rate_m3_s < 0:
        logging.warning("Received negative flow rate demand: %.3f m^3/s",
                        flow_rate_m3_s)
      return # No change in demand for non-positive flow rate

    self._air_flow_rate += flow_rate_m3_s
    # Cap at maximum flow rate
    if self._air_flow_rate > self.max_air_flow_rate:
      self._air_flow_rate = self.max_air_flow_rate
    self._cooling_request_count += 1 # Interpreted as general demand count

  def compute_thermal_energy_rate(
      self, recirculation_temp_k: float, ambient_temp_k: float
  ) -> float:
    """Calculates the thermal power (W) for heating or cooling mixed air.

    Positive values indicate heating, negative values indicate cooling.

    Args:
      recirculation_temp_k (float): Temperature (K) of recirculated air.
      ambient_temp_k (float): Temperature (K) of outdoor air.

    Returns:
      float: Thermal power (Watts) required. Positive for heating, negative
      for cooling, zero if no conditioning is needed.
    """
    mixed_air_temp_k = self.get_mixed_air_temp(
        recirculation_temp_k, ambient_temp_k
    )
    supply_air_temp_k = self.get_supply_air_temp(
        recirculation_temp_k, ambient_temp_k
    )
    # Q = m_dot * C_p * delta_T
    # m_dot = rho * V_dot (air_flow_rate is V_dot)
    # Using AIR_DENSITY and AIR_HEAT_CAPACITY from constants
    mass_flow_rate_kg_s = self._air_flow_rate * constants.AIR_DENSITY
    thermal_power_watts = (
        mass_flow_rate_kg_s *
        constants.AIR_HEAT_CAPACITY *
        (supply_air_temp_k - mixed_air_temp_k)
    )
    return thermal_power_watts

  def compute_fan_power(
      self,
      flow_rate_m3_s: float,
      fan_differential_pressure_pa: float,
      fan_efficiency_ratio: float,
  ) -> float:
    """Calculates the electrical power (W) consumed by a fan.

    Formula based on: P = (V_dot * delta_P) / efficiency
    Ref: https://www.engineeringtoolbox.com/fans-efficiency-power-consumption-d_197.html

    Args:
      flow_rate_m3_s (float): Volumetric air flow rate (m^3/s) through the fan.
      fan_differential_pressure_pa (float): Pressure difference (Pascals)
        across the fan.
      fan_efficiency_ratio (float): Electrical efficiency of the fan (0.0-1.0).

    Returns:
      float: Electrical power (Watts) consumed by the fan. Returns 0.0 if
      efficiency is zero to prevent division by zero.
    """
    if fan_efficiency_ratio == 0:
      return 0.0
    return (flow_rate_m3_s * fan_differential_pressure_pa) / fan_efficiency_ratio

  def compute_intake_fan_energy_rate(self) -> float:
    """Calculates electrical power (W) consumed by the supply/intake fan."""
    return self.compute_fan_power(
        flow_rate_m3_s=self._air_flow_rate, # Uses total AHU flow rate
        fan_differential_pressure_pa=self._fan_differential_pressure,
        fan_efficiency_ratio=self._fan_efficiency,
    )

  def compute_exhaust_fan_energy_rate(self) -> float:
    """Calculates electrical power (W) consumed by the exhaust/return fan.

    Assumes exhaust fan handles only the fresh air portion if recirculation is
    used, or total flow if no recirculation (though this might vary by AHU
    design). The current implementation uses `ambient_flow_rate`.
    """
    return self.compute_fan_power(
        flow_rate_m3_s=self.ambient_flow_rate, # Flow rate of non-recirculated air
        fan_differential_pressure_pa=self._fan_differential_pressure,
        fan_efficiency_ratio=self._fan_efficiency,
    )
