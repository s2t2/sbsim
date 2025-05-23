"""Utility functions for energy calculations in smart building simulations.

This module provides a collection of functions for calculating various
energy-related quantities relevant to HVAC systems and building thermodynamics.
These include:
- Water vapor partial pressure and humidity ratio calculations.
- Air conditioning energy rate estimation.
- Fan power and volumetric flow rate calculations.
- Compressor power estimation using thermal and utilization-based methods.
- Water pump power and volumetric flow rate calculations.
- Water heating energy rate calculations for different system configurations
  (simple, primary, primary-secondary).

Many of these functions are based on standard thermodynamic equations and
engineering principles, often citing external sources like engineering handbooks
or specific energy calculation methodologies (e.g., "go/sb-energy-calculations").
"""

from typing import Optional, Sequence

import numpy as np

from smart_control.utils import constants

# Reference data for water vapor saturation pressure.
# Source: Thermodynamik, (1992), Hans Dieter Baehr, 8. Auflage, Springer Verlag
# Tabelle 5.4, p. 213. Temperatures are in Kelvin, pressures in mbar.
_WATER_SATURATION_TEMPS_REF_K = np.array(
    [i + 273.15 for i in range(-40, 81, 10)] # Corrected range to include 80
)
_WATER_SATURATION_PRESSURES_REF_MBAR = np.array([
    0.1285, 0.3802, 1.0328, 2.5992, 6.1115, 12.279, 23.385, 42.452,
    73.813, 123.448, 199.33, 311.77, 473.9 # Added value for 80C if available
])
# Ensure arrays have same length if data for 80C is not from the exact source.
# If _WATER_SATURATION_PRESSURES_REF has 12 elements, _WATER_SATURATION_TEMPS_REF should match.
# The original range(-40, 80, 10) produces 12 values.
# If table 5.4 has a value for 80C, it should be included.
# For now, assuming the original arrays were matched. Let's stick to 12 values
# if the last value for 80C is not directly from the source.
# Corrected range based on original code: range(-40, 80, 10) yields 12 values.
# So, the pressure array should also have 12 values.
_WATER_SATURATION_TEMPS_REF_K = np.array(
    [i + 273.15 for i in range(-40, 80, 10)]
)
_WATER_SATURATION_PRESSURES_REF_MBAR = np.array([
    0.1285, 0.3802, 1.0328, 2.5992, 6.1115, 12.279, 23.385, 42.452,
    73.813, 123.448, 199.33, 311.77
])


# Operational thresholds for HVAC components
_FAN_SPEED_PERCENTAGE_OPERATIONAL_THRESH: float = 5.0
_SUPPLY_STATIC_PRESSURE_OPERATIONAL_THRESH_PSI: float = 0.2 # in psi
_DEFAULT_EER_BTU_PER_WH: float = 12.0 # Energy Efficiency Ratio in BTU/Wh


def get_water_vapor_partial_pressure(
    temperatures_k: Sequence[float]
) -> np.ndarray:
  """Calculates water vapor partial pressure in moist air using interpolation.

  Uses a lookup table of saturation temperatures and pressures from a
  thermodynamics reference.

  Args:
    temperatures_k (Sequence[float]): A sequence of air temperatures in Kelvin.

  Returns:
    np.ndarray: An array of corresponding water vapor partial pressures in
    millibars (mbar).
  """
  return np.interp(
      temperatures_k,
      _WATER_SATURATION_TEMPS_REF_K,
      _WATER_SATURATION_PRESSURES_REF_MBAR
  )


def get_humidity_ratio(
    temperatures_k: Sequence[float],
    relative_humidities: Sequence[float],
    pressures_bar: Sequence[float],
) -> np.ndarray:
  """Calculates the humidity ratio (mass of water vapor per mass of dry air).

  Formula based on: Thermodynamik, (1992), Hans Dieter Baehr, Gleichung 5.26.
  Humidity Ratio (x) = 0.622 * (phi * p_sat) / (p_atm - phi * p_sat)
  where:
    phi = relative humidity
    p_sat = saturation vapor pressure of water
    p_atm = atmospheric pressure

  Args:
    temperatures_k (Sequence[float]): Air temperatures in Kelvin.
    relative_humidities (Sequence[float]): Relative humidities (0.0 to 1.0).
    pressures_bar (Sequence[float]): Atmospheric pressures in bar.

  Returns:
    np.ndarray: An array of humidity ratios (kg water vapor / kg dry air).

  Raises:
    AssertionError: If input sequences have different lengths.
  """
  if not (len(temperatures_k) == len(relative_humidities) == len(pressures_bar)):
    raise AssertionError("All input sequences must have the same length.")

  # Saturation pressure from mbar to bar for consistency with atmospheric pressure
  saturation_pressures_bar = get_water_vapor_partial_pressure(temperatures_k) / 1000.0

  humidity_ratios = np.zeros_like(temperatures_k, dtype=float)
  for i in range(len(temperatures_k)):
    phi_p_sat = relative_humidities[i] * saturation_pressures_bar[i]
    denominator = pressures_bar[i] - phi_p_sat
    if denominator <= 0: # Avoid division by zero or negative (non-physical)
        # This can happen if phi_p_sat >= p_atm, e.g. 100% RH at boiling point
        # or incorrect pressure data. For psychrometric calcs, p_atm > p_vapor.
        # A very large humidity ratio would result. Capping or error handling needed.
        # For now, let's assume it implies very high humidity, but avoid error.
        # Or, if pressure_bar is too low.
        humidity_ratios[i] = np.nan # Or some indicator of invalid input
    else:
        humidity_ratios[i] = 0.622 * phi_p_sat / denominator
  return humidity_ratios


def get_air_conditioning_energy_rate(
    *, # Enforce keyword arguments
    air_mass_flow_rates_kg_s: Sequence[float],
    outside_temps_k: Sequence[float],
    outside_relative_humidities: Sequence[float],
    supply_temps_k: Sequence[float],
    ambient_pressures_bar: Sequence[float],
) -> np.ndarray:
  """Calculates thermal power (W) for conditioning moist outdoor air.

  This function determines the energy rate required to change the temperature
  of moist air from its outside state to the desired supply air temperature.
  It considers the enthalpy change of both dry air and water vapor.
  Assumes isobaric process, no additional (de)humidification beyond what's
  implied by temperature change, and non-saturated outside air.

  Formula based on: Q = m_dot_air * (h_supply - h_outside)
                  h_moist_air approx C_p_dry_air * T + x * C_p_vapor * T
  So, Q = m_dot_air * [ (C_p_dry_air + x * C_p_vapor) * (T_supply - T_outside) ]
  (assuming humidity ratio 'x' remains constant, which is a simplification).

  Args:
    air_mass_flow_rates_kg_s (Sequence[float]): Mass flow rates of outside
      air in kg/s.
    outside_temps_k (Sequence[float]): Outdoor air temperatures in Kelvin.
    outside_relative_humidities (Sequence[float]): Outdoor air relative
      humidities (0.0 to 1.0).
    supply_temps_k (Sequence[float]): Target supply air temperatures in Kelvin.
    ambient_pressures_bar (Sequence[float]): Ambient atmospheric pressures in bar.

  Returns:
    np.ndarray: An array of thermal power values (Watts) required. Positive
    for heating, negative for cooling.
  """
  num_samples = len(air_mass_flow_rates_kg_s)
  if not (num_samples == len(outside_temps_k) ==
            len(outside_relative_humidities) == len(supply_temps_k) ==
            len(ambient_pressures_bar)):
    raise AssertionError("All input sequences must have the same length.")

  humidity_ratios_x = get_humidity_ratio(
      temps=outside_temps_k,
      relative_humidities=outside_relative_humidities,
      pressures=ambient_pressures_bar,
  )

  energy_rates_watts = np.zeros(num_samples, dtype=float)
  for i in range(num_samples):
    # Effective specific heat of moist air (J/kg_dry_air K)
    cp_moist_air = (
        constants.AIR_HEAT_CAPACITY +
        humidity_ratios_x[i] * constants.WATER_VAPOR_HEAT_CAPACITY
    )
    delta_temp_k = supply_temps_k[i] - outside_temps_k[i]
    energy_rates_watts[i] = air_mass_flow_rates_kg_s[i] * cp_moist_air * delta_temp_k
  return energy_rates_watts


def get_fan_power(
    *, # Enforce keyword arguments
    design_hp: Optional[float] = None,
    brake_hp: Optional[float] = None,
    fan_speed_percentage: Optional[float] = None, # Range 0-100
    supply_static_pressure_psi: Optional[float] = None,
    motor_factor: float = 0.85, # Default efficiency factor
    num_fans: int = 1,
) -> float:
  """Estimates fan power consumption in Watts.

  Calculation is based on motor horsepower (design or brake), fan speed,
  and operational status inferred from static pressure or fan speed.
  This method uses empirical formulas and rules of thumb common in HVAC energy
  estimation.

  Args:
    design_hp (Optional[float]): Design horsepower of the fan motor.
    brake_hp (Optional[float]): Brake horsepower (horsepower delivered to the
      fan shaft). If provided, `motor_factor` is ignored.
    fan_speed_percentage (Optional[float]): Current operating speed of the fan
      as a percentage of its maximum speed (0-100). Defaults to 100.0 if None.
    supply_static_pressure_psi (Optional[float]): Static pressure (in PSI) at
      the fan outlet. Used to infer if the supply fan is operational.
    motor_factor (float): Efficiency factor applied to `design_hp` to estimate
      brake horsepower if `brake_hp` is not directly provided. Represents
      combined motor and drive efficiency. Default is 0.85.
    num_fans (int): Number of identical fans operating with these parameters.

  Returns:
    float: Estimated fan power consumption in Watts.

  Raises:
    ValueError: If neither `design_hp` nor `brake_hp` is provided.
  """
  if design_hp is None and brake_hp is None:
    raise ValueError("Either design_hp or brake_hp must be provided.")

  effective_fan_speed_percentage = fan_speed_percentage if fan_speed_percentage is not None else 100.0

  # Determine horsepower at the shaft
  shaft_hp = brake_hp if brake_hp is not None else (design_hp * motor_factor if design_hp is not None else 0)


  # Determine if fan is considered operational
  is_operational_flag: float = 1.0
  if supply_static_pressure_psi is not None: # Primarily for supply fans
    if supply_static_pressure_psi < _SUPPLY_STATIC_PRESSURE_OPERATIONAL_THRESH_PSI:
      is_operational_flag = 0.0
  elif fan_speed_percentage is not None: # For exhaust or other fans
    if effective_fan_speed_percentage < _FAN_SPEED_PERCENTAGE_OPERATIONAL_THRESH:
      is_operational_flag = 0.0
  # If neither pressure nor speed is given, it defaults to operational (speed 100%)

  # Fan power law: Power proportional to (speed_ratio)^3, but often (speed_ratio)^2.5 used.
  # P = P_design * (speed_actual / speed_design)^2.5 (approx.)
  # 0.746 kW per HP
  power_watts = (
      shaft_hp *
      0.746 * constants.KW_TO_W * # Convert HP to Watts
      (effective_fan_speed_percentage / 100.0)**2.5 *
      is_operational_flag *
      float(num_fans)
  )
  return power_watts


def get_air_volumetric_flowrate(
    *, # Enforce keyword arguments
    average_fan_speed_percentage: float, # Range 0-100
    design_cfm: float
) -> float:
  """Calculates AHU volumetric air flow rate based on fan speed and design CFM.

  Args:
    average_fan_speed_percentage (float): Average operating speed of the fan(s)
      as a percentage of maximum (0-100).
    design_cfm (float): Design volumetric air flow rate of the AHU in cubic
      feet per minute (CFM) at 100% fan speed.

  Returns:
    float: Estimated current volumetric air flow rate in CFM.
  """
  return design_cfm * (average_fan_speed_percentage / 100.0)


def get_compressor_power_thermal(
    *, # Enforce keyword arguments
    mixed_air_temp_f: float,
    supply_air_temp_f: float,
    volumetric_flow_rate_cfm: float,
    fan_speed_percentage: float = 100.0, # Range 0-100
    eer_btu_per_wh: float = _DEFAULT_EER_BTU_PER_WH,
    fan_heat_gain_f: float = 0.0, # Temp rise in °F due to fan heat
) -> float:
  """Estimates AC compressor power (kW) using the thermal method.

  This method calculates cooling load based on air properties and then uses
  the Energy Efficiency Ratio (EER) to estimate compressor power.
  Formula: Power (kW) = (1.08 * CFM * delta_T_F) / 12000 * (12 / EER)
  where 1.08 is a factor for air density and specific heat at standard cond.
  12000 BTU/hr = 1 Ton of refrigeration.

  Args:
    mixed_air_temp_f (float): Temperature (°F) of air entering the cooling coil.
    supply_air_temp_f (float): Temperature (°F) of air leaving the cooling coil.
    volumetric_flow_rate_cfm (float): Air flow rate in CFM.
    fan_speed_percentage (float): Fan speed (0-100). Used to determine if
      the system is operational. Defaults to 100.0.
    eer_btu_per_wh (float): Energy Efficiency Ratio of the AC unit in BTU/Wh.
      Defaults to `_DEFAULT_EER_BTU_PER_WH`.
    fan_heat_gain_f (float): Estimated temperature increase (°F) across the
      supply fan due to motor heat, which adds to the cooling load. Defaults to 0.

  Returns:
    float: Estimated compressor power consumption in kilowatts (kW).
  """
  is_fan_operational = 1.0 if fan_speed_percentage >= \
                       _FAN_SPEED_PERCENTAGE_OPERATIONAL_THRESH else 0.0

  if is_fan_operational == 0.0 or eer_btu_per_wh == 0:
    return 0.0

  # delta_T should be positive for cooling load calculation
  delta_t_cooling_f = mixed_air_temp_f - supply_air_temp_f + fan_heat_gain_f
  if delta_t_cooling_f <= 0: # No cooling load if supply is not cooler
      return 0.0

  # kW/Ton efficiency = 12 / EER (BTU/Wh)
  kw_per_ton_efficiency = 12.0 / eer_btu_per_wh

  # Cooling load in BTU/hr = 1.08 * CFM * delta_T_F
  # Cooling load in Tons = (1.08 * CFM * delta_T_F) / 12000
  compressor_power_kw = (
      1.08 * volumetric_flow_rate_cfm * delta_t_cooling_f / 12000.0
  ) * kw_per_ton_efficiency
  return compressor_power_kw


def get_compressor_power_utilization(
    *, # Enforce keyword arguments
    design_capacity_tons: float,
    cooling_percentage: Optional[float] = None, # Range 0-100
    count_stages_on: Optional[int] = None,
    total_stages: Optional[int] = None,
    eer_btu_per_wh: Optional[float] = None, # BTU/Wh
) -> float:
  """Estimates AC compressor power (kW) based on utilization ratio.

  This method uses the compressor's design capacity and its current utilization
  (either as a direct percentage or from number of active stages) along with
  its EER to estimate power.

  Args:
    design_capacity_tons (float): Total design cooling capacity of the
      compressor in tons of refrigeration.
    cooling_percentage (Optional[float]): Current cooling output as a
      percentage of design capacity (0-100). Preferred if available.
    count_stages_on (Optional[int]): Number of active cooling stages. Used if
      `cooling_percentage` is None.
    total_stages (Optional[int]): Total number of cooling stages. Required if
      `count_stages_on` is used.
    eer_btu_per_wh (Optional[float]): Energy Efficiency Ratio of the AC unit.
      Defaults to `_DEFAULT_EER_BTU_PER_WH` if None.

  Returns:
    float: Estimated compressor power consumption in kilowatts (kW).

  Raises:
    ValueError: If inputs are insufficient or invalid (e.g., negative stages,
      `cooling_percentage` out of range).
  """
  effective_eer = eer_btu_per_wh if eer_btu_per_wh is not None else _DEFAULT_EER_BTU_PER_WH
  if effective_eer == 0: return 0.0 # Avoid division by zero

  utilization_ratio: float
  if cooling_percentage is not None:
    if not (0.0 <= cooling_percentage <= 100.0):
      raise ValueError("cooling_percentage must be between 0 and 100.")
    utilization_ratio = cooling_percentage / 100.0
  elif total_stages is not None and count_stages_on is not None:
    if total_stages <= 0:
      raise ValueError("total_stages must be positive.")
    if not (0 <= count_stages_on <= total_stages):
      raise ValueError("count_stages_on must be between 0 and total_stages.")
    utilization_ratio = float(count_stages_on) / float(total_stages)
  else:
    raise ValueError(
        "Either cooling_percentage or (count_stages_on and total_stages) "
        "must be provided."
    )

  # kW/Ton efficiency = 12 / EER (BTU/Wh)
  kw_per_ton_efficiency = 12.0 / effective_eer
  return utilization_ratio * design_capacity_tons * kw_per_ton_efficiency


def get_water_pump_power(
    *, # Enforce keyword arguments
    pump_duty_cycle: float, # Range 0.0-1.0
    pump_speed_percentage: float = 100.0, # Range 0-100
    brake_hp: Optional[float] = None,
    design_motor_hp: Optional[float] = None,
    motor_factor: float = 0.85, # Combined motor & drive efficiency
    num_pumps: int = 1,
) -> float:
  """Calculates hot water pump power consumption in kilowatts (kW).

  Estimates power based on motor horsepower (design or brake), pump speed,
  duty cycle, and number of pumps.

  Args:
    pump_duty_cycle (float): Fraction of time the pump is running (0.0 to 1.0).
    pump_speed_percentage (float): Operating speed as a percentage of maximum
      (0-100). Defaults to 100.0.
    brake_hp (Optional[float]): Brake horsepower (power delivered to pump shaft).
      If provided, `motor_factor` is ignored.
    design_motor_hp (Optional[float]): Design horsepower of the pump motor.
      Used if `brake_hp` is not provided.
    motor_factor (float): Efficiency factor applied to `design_motor_hp`.
      Defaults to 0.85.
    num_pumps (int): Number of identical pumps operating. Defaults to 1.

  Returns:
    float: Estimated pump power in kilowatts (kW).

  Raises:
    ValueError: If neither `brake_hp` nor `design_motor_hp` is provided.
  """
  if brake_hp is None and design_motor_hp is None:
    raise ValueError("Either brake_hp or design_motor_hp must be provided.")

  shaft_hp = brake_hp if brake_hp is not None else \
             (design_motor_hp * motor_factor if design_motor_hp is not None else 0.0)

  # Power (kW) = HP * 0.746 * (speed_ratio)^2.5 * duty_cycle * num_pumps
  # Fan/pump affinity laws suggest P ~ speed^3, but 2.5 is often used empirically.
  power_kw = (
      shaft_hp *
      0.746 * # HP to kW conversion
      (pump_speed_percentage / 100.0)**2.5 *
      pump_duty_cycle *
      float(num_pumps)
  )
  return power_kw


def get_water_volumetric_flow_rate(
    *, # Enforce keyword arguments
    design_flow_rate_gpm: float,
    pump_speed_percentage: float, # Range 0-100
    num_pumps_on: int = 1,
) -> float:
  """Calculates water pump volumetric flow rate in gallons per minute (GPM).

  Args:
    design_flow_rate_gpm (float): Design flow rate of a single pump in GPM
      at 100% speed.
    pump_speed_percentage (float): Average operating speed of pumps as a
      percentage of maximum (0-100).
    num_pumps_on (int): Number of identical pumps currently running.

  Returns:
    float: Total estimated water volumetric flow rate in GPM.
  """
  return float(num_pumps_on) * (pump_speed_percentage / 100.0) * design_flow_rate_gpm


def get_water_heating_energy_rate(
    *, # Enforce keyword arguments
    volumetric_flow_rate_gpm: float,
    supply_water_temp_f: float,
    return_water_temp_f: float,
) -> float:
  """Computes water heating energy rate in BTU/hr for a simple loop.

  Formula: Q (BTU/hr) = 500 * GPM * delta_T_F
  The factor 500 = 8.34 (lb/gal) * 60 (min/hr) * 1 (BTU/lb°F for water).

  Args:
    volumetric_flow_rate_gpm (float): Water flow rate in GPM.
    supply_water_temp_f (float): Temperature (°F) of water supplied (heated).
    return_water_temp_f (float): Temperature (°F) of water returning (cooler).

  Returns:
    float: Heating energy rate in BTU/hr. Returns 0 if supply temperature is
    not greater than return temperature.
  """
  delta_t_f = supply_water_temp_f - return_water_temp_f
  if delta_t_f <= 0: # No heating if supply is not hotter than return
    return 0.0
  return 500.0 * volumetric_flow_rate_gpm * delta_t_f


def get_water_heating_energy_rate_primary(
    *, # Enforce keyword arguments
    design_boiler_flow_rate_gpm: float,
    boiler_outlet_temp_f: float,
    system_return_water_temp_f: float,
    num_active_boilers: int = 1,
) -> float:
  """Computes heating rate (BTU/hr) for a primary boiler loop.

  Used for systems where boilers directly supply the main heating loop.
  Calculates total flow from active boilers and then uses the standard water
  heating formula.

  Args:
    design_boiler_flow_rate_gpm (float): Design flow rate (GPM) of a single
      boiler's internal pump.
    boiler_outlet_temp_f (float): Temperature (°F) of water leaving the boilers.
    system_return_water_temp_f (float): Temperature (°F) of water returning
      from the building to the boilers.
    num_active_boilers (int): Number of boilers currently operating.

  Returns:
    float: Total heating energy rate from all active boilers in BTU/hr.
  """
  total_primary_flow_gpm = design_boiler_flow_rate_gpm * float(num_active_boilers)
  return get_water_heating_energy_rate(
      volumetric_flow_rate_gpm=total_primary_flow_gpm,
      supply_water_temp_f=boiler_outlet_temp_f,
      return_water_temp_f=system_return_water_temp_f,
  )


def get_water_heating_energy_rate_primary_secondary(
    *, # Enforce keyword arguments
    design_primary_boiler_flow_rate_gpm: float,
    design_secondary_pump_flow_rate_gpm: float,
    boiler_outlet_temp_f: float,
    system_return_water_temp_f: float, # From secondary loop to common pipe
    num_active_boilers: int = 1,
    num_active_secondary_pumps: int = 0,
    avg_secondary_pump_speed_percentage: float = 0.0, # Range 0-100
) -> float:
  """Computes heating rate (BTU/hr) for a primary-secondary boiler system.

  This handles scenarios where a primary loop (with boilers) is decoupled
  from a secondary loop (serving the building) by a common pipe. If primary
  flow exceeds secondary flow, some heated primary water recirculates directly
  back to the boilers, mixing with cooler system return water. This blended
  temperature becomes the effective return water temperature for the boilers.

  Args:
    design_primary_boiler_flow_rate_gpm (float): Design flow rate (GPM) of a
      single boiler's pump in the primary loop.
    design_secondary_pump_flow_rate_gpm (float): Design flow rate (GPM) of a
      single pump in the secondary loop (serving the building).
    boiler_outlet_temp_f (float): Temperature (°F) of water leaving the boilers.
    system_return_water_temp_f (float): Temperature (°F) of water returning
      from the secondary (building) loop to the common pipe.
    num_active_boilers (int): Number of boilers currently operating.
    num_active_secondary_pumps (int): Number of secondary loop pumps operating.
    avg_secondary_pump_speed_percentage (float): Average operating speed (0-100)
      of the active secondary pumps.

  Returns:
    float: Total heating energy rate from all active boilers in BTU/hr,
    considering the blended return water temperature if applicable.

  Raises:
    ValueError: If calculated primary or secondary flow rates are negative.
  """
  primary_flow_gpm = design_primary_boiler_flow_rate_gpm * float(num_active_boilers)
  if primary_flow_gpm < 0.0:
    raise ValueError("Calculated primary flow rate must be non-negative.")

  secondary_flow_gpm = (
      design_secondary_pump_flow_rate_gpm *
      float(num_active_secondary_pumps) *
      (avg_secondary_pump_speed_percentage / 100.0)
  )
  if secondary_flow_gpm < 0.0:
    raise ValueError("Calculated secondary flow rate must be non-negative.")

  # If primary flow > secondary flow, some primary water bypasses the secondary
  # loop and mixes with the return from the secondary loop.
  if primary_flow_gpm > secondary_flow_gpm and primary_flow_gpm > 0:
    bypass_flow_gpm = primary_flow_gpm - secondary_flow_gpm
    # Energy balance for mixing tee:
    # m_bypass * T_boiler_outlet + m_secondary_return * T_system_return = m_primary * T_blended_return
    blended_return_temp_f = (
        bypass_flow_gpm * boiler_outlet_temp_f +
        secondary_flow_gpm * system_return_water_temp_f
    ) / primary_flow_gpm
    effective_return_temp_f = blended_return_temp_f
  else: # All primary water goes through secondary, or no primary flow
    effective_return_temp_f = system_return_water_temp_f

  return get_water_heating_energy_rate(
      volumetric_flow_rate_gpm=primary_flow_gpm,
      supply_water_temp_f=boiler_outlet_temp_f,
      return_water_temp_f=effective_return_temp_f,
  )
