"""Models a boiler for heating water in an HVAC system.

This module defines the `Boiler` class, which simulates a central boiler that
heats water to a specified setpoint. The heated water is then typically
circulated to VAV (Variable Air Volume) units for reheating air supplied to
building zones. The model includes energy consumption for heating the water
and for the circulation pump.
"""

from typing import Optional
import uuid

import gin
import numpy as np
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import smart_device
from smart_control.utils import constants


@gin.configurable
class Boiler(smart_device.SmartDevice):
  """Simulates a central boiler with a water pump for an HVAC system.

  The boiler model maintains a supply water temperature setpoint. It calculates
  the energy required to heat incoming return water to this setpoint,
  considering thermal losses from the tank. It also models the power consumed
  by the water pump.

  Key functionalities:
  -   Setting and observing the supply water temperature setpoint.
  -   Tracking the return water temperature from the HVAC loop.
  -   Simulating the change in tank water temperature over time if heating/cooling
      rates are defined.
  -   Calculating natural gas consumption for heating.
  -   Calculating electricity consumption for the water pump.
  -   Estimating thermal losses from the boiler tank to the ambient environment.

  Attributes (selected):
    reheat_water_setpoint (float): Target temperature (K) for water supplied by
      the boiler. This is an actionable and observable field.
    supply_water_temperature_sensor (float): Current temperature (K) of the
      water supplied by the boiler. This is an observable field.
    return_water_temperature_sensor (float): Temperature (K) of water
      returning to the boiler from the HVAC loop. This is typically updated by
      the HVAC system simulation based on VAV reheat coil usage.
    heating_request_count (int): Number of downstream units (e.g., VAVs)
      currently requesting hot water.
    _total_flow_rate (float): Aggregated hot water flow rate (m^3/s) demanded
      by all connected VAVs.
    _water_pump_differential_head (float): The pressure head (meters) the pump
      works against.
    _water_pump_efficiency (float): Electrical efficiency of the water pump
      (0.0 to 1.0).
    _heating_rate (float): Rate (K/minute) at which the boiler can heat water.
    _cooling_rate (float): Rate (K/minute) at which water in the tank cools down
      if not actively heated (simplified thermal loss).
    _current_temperature (float): Internal state variable for the current
        temperature of water in the boiler tank (K).
    _step_tank_temperature_change (float): Change in tank temperature during the
        last simulation step (K).
    _last_step_duration (pd.Timedelta): Duration of the last simulation step.
    _convection_coefficient (float): Heat transfer coefficient (W/m^2K) for
      convective losses from the tank surface.
    _tank_length (float): Length (m) of the boiler tank.
    _tank_radius (float): Radius (m) of the boiler tank.
    _water_capacity (float): Water storage capacity (m^3) of the boiler tank.
    _insulation_conductivity (float): Thermal conductivity (W/mK) of the
      tank's insulation.
    _insulation_thickness (float): Thickness (m) of the tank's insulation.
  """

  def __init__(
      self,
      reheat_water_setpoint_k: float,
      water_pump_differential_head_m: float,
      water_pump_efficiency_ratio: float,
      device_id: Optional[str] = None,
      heating_rate_k_per_min: float = 0.0, # Default implies instantaneous heating
      cooling_rate_k_per_min: float = 0.0, # Default implies no passive cooling
      convection_coefficient_w_m2k: float = 5.6,
      tank_length_m: float = 2.0,
      tank_radius_m: float = 0.5,
      water_capacity_m3: float = 1.5,
      insulation_conductivity_w_mk: float = 0.067,
      insulation_thickness_m: float = 0.06,
  ):
    """Initializes the Boiler instance.

    Args:
      reheat_water_setpoint_k (float): Initial target supply water
        temperature in Kelvin.
      water_pump_differential_head_m (float): The effective pressure head
        (in meters of water column) that the pump operates against.
      water_pump_efficiency_ratio (float): Electrical efficiency of the water
        pump, as a value between 0.0 and 1.0.
      device_id (Optional[str]): A unique identifier for this boiler. If None,
        a UUID will be generated.
      heating_rate_k_per_min (float): The rate (Kelvin per minute) at which
        the boiler can increase the water temperature. If 0, heating is
        assumed to be instantaneous to the setpoint.
      cooling_rate_k_per_min (float): The rate (Kelvin per minute) at which
        the boiler water cools passively if not actively heated. If 0, passive
        cooling (other than calculated dissipation) is ignored.
      convection_coefficient_w_m2k (float): Convective heat transfer
        coefficient (W/m^2K) for calculating thermal losses from the tank surface.
      tank_length_m (float): Internal length of the boiler tank in meters.
      tank_radius_m (float): Internal radius of the boiler tank in meters.
      water_capacity_m3 (float): Total water storage capacity of the boiler
        tank in cubic meters.
      insulation_conductivity_w_mk (float): Thermal conductivity (W/mK) of the
        boiler tank's insulation material.
      insulation_thickness_m (float): Thickness (meters) of the insulation layer
        around the boiler tank.
    """
    observable_fields_info = {
        "supply_water_setpoint": smart_device.AttributeInfo(
            "reheat_water_setpoint", float # Maps to self.reheat_water_setpoint
        ),
        "supply_water_temperature_sensor": smart_device.AttributeInfo(
            "supply_water_temperature_sensor", float # Property, calls _set_current_temperature
        ),
        "heating_request_count": smart_device.AttributeInfo(
            "heating_request_count", int
        ),
    }
    action_fields_info = {
        "supply_water_setpoint": smart_device.AttributeInfo(
            "reheat_water_setpoint", float
        )
    }
    dev_id = device_id if device_id else f"boiler_id_{uuid.uuid4()}"

    super().__init__(
        observable_fields=observable_fields_info,
        action_fields=action_fields_info,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.BLR,
        device_id=dev_id,
    )

    # Store initial values for reset
    self._init_reheat_water_setpoint_k = reheat_water_setpoint_k
    self._init_pump_head_m = water_pump_differential_head_m
    self._init_pump_efficiency = water_pump_efficiency_ratio
    self._init_heating_request_count = 0
    self._init_return_water_temp_k = 0.0 # Assuming starts cold or reset

    # Boiler physical and operational parameters
    self._heating_rate_k_per_min = heating_rate_k_per_min
    self._cooling_rate_k_per_min = cooling_rate_k_per_min
    self._convection_coefficient = convection_coefficient_w_m2k
    self._tank_length = tank_length_m
    self._tank_radius = tank_radius_m
    self._water_capacity = water_capacity_m3
    self._insulation_conductivity = insulation_conductivity_w_mk
    self._insulation_thickness = insulation_thickness_m

    # State variables (will be initialized in reset)
    self._total_flow_rate: float = 0.0
    self._reheat_water_setpoint: float = 0.0
    self._water_pump_differential_head: float = 0.0
    self._water_pump_efficiency: float = 0.0
    self._heating_request_count: int = 0
    self._return_water_temperature_sensor: float = 0.0
    self._current_temperature: float = 0.0
    self._step_tank_temperature_change: float = 0.0
    self._last_step_duration: pd.Timedelta = pd.Timedelta(0, unit="s")
    self.reset()

  def reset(self) -> None:
    """Resets the boiler to its initial state."""
    self.reset_demand()
    self._reheat_water_setpoint = self._init_reheat_water_setpoint_k
    self._water_pump_differential_head = self._init_pump_head_m
    self._water_pump_efficiency = self._init_pump_efficiency
    self._heating_request_count = self._init_heating_request_count
    self._return_water_temperature_sensor = self._init_return_water_temp_k
    # Initial tank temperature is assumed to be at the initial setpoint
    self._current_temperature = self._init_reheat_water_setpoint_k
    self._step_tank_temperature_change = 0.0
    self._last_step_duration = pd.Timedelta(0, unit="s")
    logging.debug("Boiler '%s' reset.", self.device_id())


  @property
  def return_water_temperature_sensor(self) -> float:
    """Temperature (K) of water returning to the boiler."""
    return self._return_water_temperature_sensor

  @return_water_temperature_sensor.setter
  def return_water_temperature_sensor(self, value: float) -> None:
    self._return_water_temperature_sensor = value

  @property
  def reheat_water_setpoint(self) -> float:
    """Target supply water temperature (K) for the boiler."""
    return self._reheat_water_setpoint

  @reheat_water_setpoint.setter
  def reheat_water_setpoint(self, value: float) -> None:
    self._reheat_water_setpoint = value

  @property
  def heating_request_count(self) -> int:
    """Number of VAVs currently requesting hot water."""
    return self._heating_request_count

  @property
  def supply_water_temperature_sensor(self) -> float:
    """Current temperature (K) of water supplied by the boiler.

    This property dynamically updates the internal boiler water temperature
    based on elapsed time since the last setpoint change or observation,
    considering defined heating/cooling rates.
    """
    self._set_current_temperature()
    return self._current_temperature

  @property
  def supply_water_setpoint(self) -> float:
    """Alias for `reheat_water_setpoint` for observation consistency."""
    return self.reheat_water_setpoint # Delegates to the property

  def reset_demand(self) -> None:
    """Resets accumulated hot water flow rate and heating requests."""
    self._total_flow_rate = 0.0
    self._heating_request_count = 0

  def _set_current_temperature(self) -> None:
    """Updates internal current water temperature based on time elapsed.

    If `_heating_rate_k_per_min` or `_cooling_rate_k_per_min` are non-zero,
    this method simulates the gradual temperature change towards the setpoint.
    Otherwise, the temperature is assumed to instantaneously match the setpoint.
    This method is called by `supply_water_temperature_sensor` property.
    """
    # If _action_timestamp (when setpoint was last changed) is available,
    # calculate duration since then.
    if self._action_timestamp: # Set by SmartDevice base class on action
      self._last_step_duration = (
          self._observation_timestamp - self._action_timestamp # Timestamps from SmartDevice
      )
    else: # No action yet in this episode, assume observation_timestamp is start
      self._action_timestamp = self._observation_timestamp
      self._last_step_duration = pd.Timedelta(0, unit="s")

    # Only adjust if rates are defined and time has passed
    if (self._last_step_duration.total_seconds() > 0 and
        (self._cooling_rate_k_per_min > 0.0 or self._heating_rate_k_per_min > 0.0)):
      previous_temp_k = self._current_temperature
      self._current_temperature = self._adjust_temperature(
          self._reheat_water_setpoint, previous_temp_k, self._last_step_duration
      )
      self._step_tank_temperature_change = self._current_temperature - previous_temp_k
    else: # Instantaneous or no time elapsed
      self._current_temperature = self._reheat_water_setpoint
      self._step_tank_temperature_change = 0.0


  def _adjust_temperature(
      self,
      setpoint_temp_k: float,
      current_temp_k: float,
      time_delta: pd.Timedelta,
  ) -> float:
    """Linearly adjusts temperature towards setpoint based on heating/cooling rates.

    Args:
      setpoint_temp_k (float): The target temperature (K).
      current_temp_k (float): The current water temperature (K) in the tank.
      time_delta (pd.Timedelta): The duration over which the change occurs.

    Returns:
      float: The new water temperature (K) after considering heating/cooling
      over `time_delta`, capped by the `setpoint_temp_k`.
    """
    delta_seconds = time_delta.total_seconds()
    if delta_seconds <= 0:
      return current_temp_k

    if setpoint_temp_k > current_temp_k: # Heating needed
      max_increase_k = self._heating_rate_k_per_min * (delta_seconds / 60.0)
      new_temp_k = current_temp_k + max_increase_k
      return min(new_temp_k, setpoint_temp_k)
    elif setpoint_temp_k < current_temp_k: # Cooling needed (passive or active)
      max_decrease_k = self._cooling_rate_k_per_min * (delta_seconds / 60.0)
      new_temp_k = current_temp_k - max_decrease_k
      return max(new_temp_k, setpoint_temp_k)
    else: # Already at setpoint
      return setpoint_temp_k

  def add_demand(self, flow_rate_m3_s: float) -> None:
    """Adds to the current hot water flow rate demand from VAVs.

    Args:
      flow_rate_m3_s (float): Hot water flow rate (m^3/s) to add.

    Raises:
      ValueError: If `flow_rate_m3_s` is not positive.
    """
    if flow_rate_m3_s <= 0:
      # Allow zero flow if VAV is closed, but log if negative.
      if flow_rate_m3_s < 0:
          logging.warning("Received negative hot water flow demand: %.3f m^3/s",
                          flow_rate_m3_s)
      return
    self._total_flow_rate += flow_rate_m3_s
    self._heating_request_count += 1

  def compute_thermal_energy_rate(
      self, return_water_temp_k: float, ambient_temp_k: float
  ) -> float:
    """Calculates total thermal power (W) consumed by the boiler.

    This includes:
    1.  Energy to heat the circulated water flow from `return_water_temp_k` to
        the `supply_water_temperature_sensor` (current tank temperature).
    2.  Energy to compensate for thermal dissipation (losses) from the tank.
    3.  Energy required to change the temperature of the water stored in the
        tank itself during the last time step.

    Args:
      return_water_temp_k (float): Temperature (K) of water returning from VAVs.
      ambient_temp_k (float): Ambient temperature (K) surrounding the boiler tank.

    Returns:
      float: Total thermal power (Watts) consumed by the boiler.
    """
    # Ensure current supply temperature is up-to-date
    current_supply_temp_k = self.supply_water_temperature_sensor

    # Energy to heat the water flowing through the boiler
    # If return water is hotter than supply (e.g. due to setpoint change),
    # this term could be negative, meaning no heating needed for flow.
    # However, boiler typically only adds heat.
    delta_temp_flow_k = max(0, current_supply_temp_k - return_water_temp_k)
    flow_heating_power_w = (
        constants.WATER_DENSITY * # kg/m^3
        self._total_flow_rate *   # m^3/s
        constants.WATER_SPECIFIC_HEAT_J_KGK * # J/kgK
        delta_temp_flow_k        # K
    )

    # Energy to compensate for tank thermal losses
    dissipation_power_w = self.compute_thermal_dissipation_rate(
        current_supply_temp_k, ambient_temp_k
    )

    # Energy to change temperature of water stored in the tank
    tank_heating_power_w: float = 0.0
    if self._last_step_duration.total_seconds() > 0:
      tank_mass_kg = self._water_capacity * constants.WATER_DENSITY
      tank_heating_power_w = (
          tank_mass_kg *
          constants.WATER_SPECIFIC_HEAT_J_KGK *
          self._step_tank_temperature_change / # K
          self._last_step_duration.total_seconds() # s
      )
    # Ensure tank heating power is non-negative as boiler only adds heat
    tank_heating_power_w = max(0, tank_heating_power_w)


    # Total power is sum of heating flow, compensating losses, and heating tank.
    # If the setpoint was lowered and tank passively cooled, tank_heating_power_w
    # would be based on _step_tank_temperature_change which could be negative.
    # However, a real boiler only consumes gas to *add* heat.
    # The current logic correctly sums positive contributions.
    return flow_heating_power_w + dissipation_power_w + tank_heating_power_w

  def compute_thermal_dissipation_rate(
      self, water_temp_k: float, ambient_temp_k: float
  ) -> float:
    """Calculates thermal power loss (W) from the boiler tank to ambient.

    Models heat loss through cylindrical tank walls considering conduction
    through insulation and convection from the outer surface. End cap losses
    are neglected.

    The heat transfer `Q` is modeled by:
    `Q = (T_water - T_ambient) / R_total`
    where `R_total = R_conduction + R_convection`.
    `R_conduction = ln(r_outer/r_inner) / (2 * pi * L * k_insulation)`
    `R_convection = 1 / (h_conv * A_outer)`
    `A_outer = 2 * pi * r_outer * L`

    Args:
      water_temp_k (float): Average temperature (K) of water inside the tank.
      ambient_temp_k (float): Temperature (K) of the air surrounding the tank.

    Returns:
      float: Rate of thermal energy loss (Watts) from the tank. Returns 0 if
      water temperature is not higher than ambient.
    """
    if water_temp_k <= ambient_temp_k:
      return 0.0 # No heat loss if water is not hotter than ambient

    delta_temp_k = water_temp_k - ambient_temp_k
    r_inner_m = self._tank_radius
    r_outer_m = r_inner_m + self._insulation_thickness

    # Thermal resistance due to conduction through insulation
    # (avoid division by zero if r_outer == r_inner or k_ins == 0)
    if r_outer_m <= r_inner_m or self._insulation_conductivity == 0:
      r_conduction_K_W = float("inf") # Effectively infinite resistance
    else:
      r_conduction_K_W = np.log(r_outer_m / r_inner_m) / (
          2 * np.pi * self._tank_length * self._insulation_conductivity
      )

    # Thermal resistance due to convection from outer surface
    # (avoid division by zero if h_conv == 0 or r_outer == 0)
    if self._convection_coefficient == 0 or r_outer_m == 0:
      r_convection_K_W = float("inf")
    else:
      outer_surface_area_m2 = 2 * np.pi * r_outer_m * self._tank_length
      r_convection_K_W = 1.0 / (self._convection_coefficient * outer_surface_area_m2)

    total_thermal_resistance_K_W = r_conduction_K_W + r_convection_K_W
    if total_thermal_resistance_K_W == 0 or np.isinf(total_thermal_resistance_K_W):
        return 0.0 # Avoid division by zero or handle infinite resistance

    dissipation_rate_W = delta_temp_k / total_thermal_resistance_K_W
    return dissipation_rate_W

  def compute_pump_power(self) -> float:
    """Calculates electrical power (W) consumed by the hot water pump.

    Formula: `P = (rho * g * H * Q) / eta`
    where:
      P = Power (Watts)
      rho = Water density (kg/m^3)
      g = Gravitational acceleration (m/s^2)
      H = Pump head (meters)
      Q = Volumetric flow rate (m^3/s)
      eta = Pump efficiency (ratio)

    Ref: https://www.engineeringtoolbox.com/pumps-power-d_505.html
    """
    if self._water_pump_efficiency == 0:
      return 0.0 # Avoid division by zero for zero efficiency

    power_watts = (
        self._total_flow_rate *             # Q (m^3/s)
        constants.WATER_DENSITY *           # rho (kg/m^3)
        constants.GRAVITY *                 # g (m/s^2)
        self._water_pump_differential_head / # H (m)
        self._water_pump_efficiency         # eta (dimensionless)
    )
    return power_watts
