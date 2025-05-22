"""Simulation model for a boiler in an HVAC system.

This module provides the `Boiler` class, which simulates the operation of a
central boiler. This includes heating water to a specified setpoint, tracking
thermal losses from the boiler tank, aggregating hot water demand, and calculating
the energy consumption of both the heating element (e.g., gas burner) and the
water circulation pump.
"""

from typing import Optional, Dict # Added Dict for type hint
import uuid

import gin
import numpy as np
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import smart_device
from smart_control.utils import constants


@gin.configurable
class Boiler(smart_device.SmartDevice):
  """Models a central boiler with a water pump for heating applications.

  This class simulates the boiler's thermal behavior, including maintaining a
  supply water temperature setpoint, responding to heating demands, and accounting
  for heat loss from the tank. It also calculates the energy consumed by the
  heating element (e.g., natural gas) and the electrical energy for the pump.
  It inherits from `SmartDevice` to define its observable and actionable
  properties.
  """

  def __init__(
      self,
      reheat_water_setpoint: float,
      water_pump_differential_head: float,
      water_pump_efficiency: float,
      device_id: Optional[str] = None,
      heating_rate: Optional[float] = 0.0, # Default to 0 for clarity
      cooling_rate: Optional[float] = 0.0, # Default to 0 for clarity
      convection_coefficient: Optional[float] = 5.6,
      tank_length: Optional[float] = 2.0,
      tank_radius: Optional[float] = 0.5,
      water_capacity: Optional[float] = 1.5,
      insulation_conductivity: Optional[float] = 0.067,
      insulation_thickness: Optional[float] = 0.06,
  ):
    """Initializes the Boiler instance.

    Args:
      reheat_water_setpoint: The target temperature (in Kelvin, K) that the
        boiler aims to maintain for the supply water.
      water_pump_differential_head: The pressure head (in meters, m) that the
        water pump operates against to circulate water.
      water_pump_efficiency: The electrical efficiency of the water pump, as a
        dimensionless ratio between 0.0 (no efficiency) and 1.0 (perfect
        efficiency).
      device_id: An optional unique string identifier for this boiler. If None,
        a random UUID will be generated.
      heating_rate: The rate (in degrees Celsius or Kelvin per minute) at which
        the boiler can heat its water volume when actively heating. If 0,
        temperature changes might be modeled as instantaneous or based solely
        on other thermal calculations. Defaults to 0.0.
      cooling_rate: The natural cooling rate (in degrees Celsius or Kelvin per
        minute) of the water in the boiler tank due to ambient losses when not
        actively heating (this is separate from calculated thermal dissipation
        through insulation). If 0, this specific cooling dynamic is not applied.
        Defaults to 0.0.
      convection_coefficient: The heat transfer coefficient (in Watts per square
        meter per Kelvin, W/m^2/K) for convective heat loss from the boiler
        tank's outer surface to the surrounding environment. Defaults to 5.6.
      tank_length: The internal length of the boiler tank (in meters, m).
        Used for thermal dissipation calculations. Defaults to 2.0.
      tank_radius: The internal radius of the boiler tank (in meters, m).
        Used for thermal dissipation calculations. Defaults to 0.5.
      water_capacity: The volume of water the boiler tank holds (in cubic
        meters, m^3). Used for calculating thermal inertia. Defaults to 1.5.
      insulation_conductivity: The thermal conductivity (in Watts per meter per
        Kelvin, W/m/K) of the boiler tank's insulation material. Defaults to 0.067.
      insulation_thickness: The thickness (in meters, m) of the boiler tank's
        insulation layer. Defaults to 0.06.
    """
    observable_fields: Dict[str, smart_device.AttributeInfo] = {
        'supply_water_setpoint': smart_device.AttributeInfo(
            internal_attribute_name='reheat_water_setpoint', attribute_type=float
        ),
        'supply_water_temperature_sensor': smart_device.AttributeInfo(
            internal_attribute_name='supply_water_temperature_sensor', attribute_type=float
        ),
        'heating_request_count': smart_device.AttributeInfo(
            internal_attribute_name='heating_request_count', attribute_type=int
        ),
    }

    action_fields: Dict[str, smart_device.AttributeInfo] = {
        'supply_water_setpoint': smart_device.AttributeInfo(
            internal_attribute_name='reheat_water_setpoint', attribute_type=float
        )
    }

    if device_id is None:
      device_id = f'boiler_id_{uuid.uuid4()}'

    super().__init__(
        observable_fields=observable_fields,
        action_fields=action_fields,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.BLR,
        device_id=device_id,
    )

    # Store initial configuration values for reset
    self._init_reheat_water_setpoint = reheat_water_setpoint
    self._init_water_pump_differential_head = water_pump_differential_head
    self._init_water_pump_efficiency = water_pump_efficiency
    self._init_heating_request_count = 0
    self._init_return_water_temperature_sensor = 0.0 # Initial return temp if no demand
    self._heating_rate = heating_rate
    self._cooling_rate = cooling_rate
    self._convection_coefficient = convection_coefficient
    self._tank_length = tank_length
    self._tank_radius = tank_radius
    self._water_capacity = water_capacity
    self._insulation_conductivity = insulation_conductivity
    self._insulation_thickness = insulation_thickness

    # Initialize state variables
    self.reset()

  def reset(self) -> None:
    """Resets the boiler's state to its initial configuration.

    This includes resetting demand counters, setpoints, and internal
    temperature states. It is typically called at the start of each new
    simulation episode.
    """
    self.reset_demand()
    self._reheat_water_setpoint = self._init_reheat_water_setpoint
    self._water_pump_differential_head = self._init_water_pump_differential_head
    self._water_pump_efficiency = self._init_water_pump_efficiency
    self._heating_request_count = self._init_heating_request_count
    self._return_water_temperature_sensor = (
        self._init_return_water_temperature_sensor
    )
    # Initialize current temperature to the setpoint at reset
    self._current_temperature = self._init_reheat_water_setpoint
    self._step_tank_temperature_change = 0.0
    self._last_step_duration = pd.Timedelta(seconds=0) # Use seconds for clarity

  @property
  def return_water_temperature_sensor(self) -> float:
    """The temperature (in Kelvin) of the water returning to the boiler."""
    return self._return_water_temperature_sensor

  @return_water_temperature_sensor.setter
  def return_water_temperature_sensor(self, value: float) -> None:
    """Sets the return water temperature (in Kelvin)."""
    self._return_water_temperature_sensor = value

  @property
  def reheat_water_setpoint(self) -> float:
    """The target supply water temperature setpoint (in Kelvin) for the boiler."""
    return self._reheat_water_setpoint

  @reheat_water_setpoint.setter
  def reheat_water_setpoint(self, value: float) -> None:
    """Sets the reheat water temperature setpoint (in Kelvin)."""
    self._reheat_water_setpoint = value

  @property
  def heating_request_count(self) -> int:
    """Number of VAVs or zones that have requested heating in the current cycle."""
    return self._heating_request_count

  @property
  def supply_water_temperature_sensor(self) -> float:
    """The current temperature (in Kelvin) of the water supplied by the boiler.

    This temperature is dynamically updated based on setpoints, heating/cooling
    rates, and elapsed time.
    """
    self._set_current_temperature()
    return self._current_temperature

  @property
  def supply_water_setpoint(self) -> float:
    """Alias for `reheat_water_setpoint` (in Kelvin), often used as an observable field."""
    return self._reheat_water_setpoint

  def reset_demand(self) -> None:
    """Resets the aggregated hot water flow rate demand and heating request count.

    This is typically called at the start of each simulation step before new
    demands are aggregated.
    """
    self._total_flow_rate = 0.0
    self._heating_request_count = 0

  def _set_current_temperature(self) -> None:
    """Updates the boiler's internal water temperature.

    This internal method adjusts `_current_temperature` based on the
    `_reheat_water_setpoint`, defined `_heating_rate` and `_cooling_rate`,
    and the time elapsed since the last action or observation.
    It also calculates `_step_tank_temperature_change` for the current step.
    """
    # If no action_timestamp is set (e.g., first step), assume it's the observation_timestamp
    if self._action_timestamp is None: # Should be initialized in SmartDevice
        self._action_timestamp = self._observation_timestamp

    if self._observation_timestamp and self._action_timestamp:
        self._last_step_duration = (
            self._observation_timestamp - self._action_timestamp
        )
    else: # Should not happen if timestamps are managed correctly
        self._last_step_duration = pd.Timedelta(seconds=0)

    if self._cooling_rate > 0.0 and self._heating_rate > 0.0:
      begin_step_temp = self._current_temperature
      self._current_temperature = self._adjust_temperature(
          self._reheat_water_setpoint, begin_step_temp, self._last_step_duration
      )
      self._step_tank_temperature_change = (
          self._current_temperature - begin_step_temp
      )
    else: # If rates are zero, assume instantaneous change to setpoint
      self._current_temperature = self._reheat_water_setpoint
      self._step_tank_temperature_change = 0.0 # No change if instantaneous

  def _adjust_temperature(
      self,
      setpoint_temperature: float,
      actual_temperature: float,
      time_difference: pd.Timedelta,
  ) -> float:
    """Linearly adjusts temperature towards setpoint based on heating/cooling rates.

    Calculates the new temperature of the boiler water after `time_difference`
    has elapsed, moving from `actual_temperature` towards `setpoint_temperature`.
    The rate of change is determined by `self._heating_rate` or
    `self._cooling_rate`.

    Args:
      setpoint_temperature: The target temperature (in Kelvin).
      actual_temperature: The current temperature of the boiler water (in Kelvin).
      time_difference: The `pd.Timedelta` duration over which the temperature
        change occurs.

    Returns:
      The new adjusted temperature in Kelvin (float), capped by the
      `setpoint_temperature`.
    """
    delta_seconds = time_difference.total_seconds()
    if setpoint_temperature > actual_temperature: # Heating needed
      increase = self._heating_rate * (delta_seconds / 60.0) # Rate is per minute
      return min(actual_temperature + increase, setpoint_temperature)
    elif setpoint_temperature < actual_temperature: # Cooling (natural loss) needed
      decrease = self._cooling_rate * (delta_seconds / 60.0) # Rate is per minute
      return max(actual_temperature - decrease, setpoint_temperature)
    else: # Already at setpoint
      return setpoint_temperature

  def add_demand(self, flow_rate: float) -> None:
    """Adds to the current hot water flow rate demand and heating request count.

    Args:
      flow_rate: The hot water flow rate demand to add (in m^3/s).

    Raises:
      ValueError: If the provided `flow_rate` is not positive.
    """
    if flow_rate <= 0:
      raise ValueError('Flow rate must be positive for boiler demand.')
    self._total_flow_rate += flow_rate
    self._heating_request_count += 1

  def compute_thermal_energy_rate(
      self, return_water_temp: float, outside_temp: float
  ) -> float:
    """Calculates the total thermal power (in Watts) required by the boiler.

    This includes:
    1.  `flow_heating_energy_rate`: Energy to heat the circulated water from
        `return_water_temp` to the `supply_water_temp` (setpoint or return temp,
        whichever is higher, as boiler doesn't cool).
    2.  `dissipation_energy_rate`: Energy to compensate for thermal losses from
        the boiler tank to the ambient `outside_temp`.
    3.  `tank_heating_energy_rate`: Energy required to change the temperature
        of the water stored within the tank itself during the last time step.

    Args:
      return_water_temp: Temperature (in Kelvin) of the water returning to the
        boiler from the heating loop.
      outside_temp: Ambient temperature (in Kelvin) surrounding the boiler tank,
        used for calculating thermal dissipation.

    Returns:
      The total thermal power requirement in Watts (float).
    """
    # Determine the effective supply water temperature for calculation.
    # Boiler only heats, so supply temp is at least the return temp.
    supply_water_temp = max(self._reheat_water_setpoint, return_water_temp)

    # 1. Energy to heat the water flowing through the system
    flow_heating_energy_rate = (
        constants.WATER_HEAT_CAPACITY * self._total_flow_rate *
        (supply_water_temp - return_water_temp)
    )
    # Ensure non-negative if return_water_temp > supply_water_temp (though logic above tries to prevent this)
    flow_heating_energy_rate = max(0.0, flow_heating_energy_rate)


    # 2. Energy to compensate for thermal dissipation from the tank
    dissipation_energy_rate = self.compute_thermal_dissipation_rate(
        water_temp=self.supply_water_temperature_sensor, # Current tank temp
        outside_temp=outside_temp
    )

    # 3. Energy to change the temperature of water stored in the tank
    tank_heating_energy_rate = 0.0
    if self._last_step_duration.total_seconds() > 0:
      # Mass of water in tank = volume (m^3) * density (kg/m^3)
      mass_of_water_in_tank = self._water_capacity * constants.WATER_DENSITY
      # Energy (J) = mass (kg) * specific_heat (J/kgK) * delta_T (K)
      energy_to_change_tank_temp = (
          mass_of_water_in_tank * constants.WATER_SPECIFIC_HEAT *
          self._step_tank_temperature_change # Change in K from _set_current_temperature
      )
      # Power (W = J/s)
      tank_heating_energy_rate = energy_to_change_tank_temp / self._last_step_duration.total_seconds()
      # Ensure non-negative as we only account for heating energy provided by boiler
      tank_heating_energy_rate = max(0.0, tank_heating_energy_rate)


    return (
        flow_heating_energy_rate + dissipation_energy_rate +
        tank_heating_energy_rate
    )

  def compute_thermal_dissipation_rate(
      self, water_temp: float, outside_temp: float
  ) -> float:
    """Calculates the thermal loss rate (Watts) from the boiler tank.

    This models heat loss from the cylindrical boiler tank to the surrounding
    environment due to imperfect insulation. It considers both conduction
    through the insulation and convection from the outer surface of the
    insulation to the ambient air. Heat loss through the tank ends/caps is ignored.

    The calculation is based on an energy balance for steady-state heat transfer
    through a cylindrical wall with internal and external convection.
    Formula: Q = (T_water - T_ambient) / R_total, where R_total is the sum of
    thermal resistances (conduction through insulation + convection from surface).
    R_conduction = ln(r_outer/r_inner) / (2 * pi * L * k_insulation)
    R_convection = 1 / (h_convection * A_outer_surface)
    A_outer_surface = 2 * pi * r_outer * L

    Args:
      water_temp: Average temperature of the water inside the tank (in Kelvin).
      outside_temp: Temperature of the environment surrounding the tank (in Kelvin).

    Returns:
      The rate of thermal energy loss from the tank in Watts (float).
      Returns 0 if `water_temp` is not greater than `outside_temp`.
    """
    if not water_temp > outside_temp:
      return 0.0

    delta_temp = water_temp - outside_temp
    
    # Geometric parameters
    r_inner = self._tank_radius
    r_outer = r_inner + self._insulation_thickness
    length = self._tank_length

    # Avoid division by zero or log of non-positive if radii are problematic
    if r_inner <= 0 or r_outer <= r_inner or self._insulation_conductivity <= 0:
        return 0.0 # Or log a warning

    # Thermal resistance of conduction through insulation
    # R_cond = ln(r_outer / r_inner) / (2 * pi * L * k)
    conduction_resistance = (
        np.log(r_outer / r_inner) /
        (2 * np.pi * length * self._insulation_conductivity)
    )

    # Thermal resistance of convection from outer surface
    # R_conv = 1 / (h * A_outer) where A_outer = 2 * pi * r_outer * L
    if self._convection_coefficient <= 0 or r_outer <= 0:
        return 0.0 # Or log a warning
        
    area_outer_surface = 2 * np.pi * r_outer * length
    convection_resistance = 1.0 / (self._convection_coefficient * area_outer_surface)
    
    total_thermal_resistance = conduction_resistance + convection_resistance
    
    if total_thermal_resistance <= 1e-9: # Avoid division by very small number
        return 0.0 # Or a very large number if appropriate, or log error

    return delta_temp / total_thermal_resistance

  def compute_pump_power(self) -> float:
    """Calculates the electrical power consumed by the water pump.

    The formula used is:
    Power (Watts) = (Flow Rate (m^3/s) * Density (kg/m^3) * Gravity (m/s^2) * Head (m)) / Efficiency
    Source: https://www.engineeringtoolbox.com/pumps-power-d_505.html

    Returns:
      The electrical power consumed by the pump in Watts (float). Returns 0.0
      if pump efficiency is zero to prevent division by zero.
    """
    if self._water_pump_efficiency == 0:
      return 0.0
    return (
        self._total_flow_rate
        * constants.WATER_DENSITY
        * constants.GRAVITY
        * self._water_pump_differential_head
        / self._water_pump_efficiency
    )
