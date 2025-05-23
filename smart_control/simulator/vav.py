"""Models a Variable Air Volume (VAV) unit for HVAC systems.

This module defines the `Vav` class, which simulates a VAV unit responsible
for controlling airflow and reheating air for a specific building zone.
Each VAV unit is typically controlled by a thermostat.
"""

from typing import Optional, Tuple
import uuid

import numpy as np # Added for np.clip
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import smart_device
from smart_control.simulator import thermostat as thermostat_py
from smart_control.utils import constants


class Vav(smart_device.SmartDevice):
  """Simulates a Variable Air Volume (VAV) unit with damper and reheat coil.

  A VAV unit controls the amount of conditioned air from an Air Handler Unit
  (AHU) that enters a zone, using a damper. It can also further heat this air
  using a reheat coil, which typically uses hot water from a boiler. The VAV's
  operation is governed by a thermostat that senses the zone temperature and
  compares it to setpoints.

  Attributes:
    max_air_flow_rate (float): Maximum air flow rate (m^3/s) when the damper
      is fully open.
    reheat_max_water_flow_rate (float): Maximum hot water flow rate (kg/s)
      through the reheat coil when its valve is fully open.
    reheat_valve_setting (float): Current setting of the reheat coil's hot
      water valve (0.0 to 1.0, where 1.0 is fully open).
    damper_setting (float): Current setting of the air damper (0.0 to 1.0,
      where 1.0 is fully open).
    thermostat (thermostat_py.Thermostat): The thermostat instance controlling
      this VAV unit.
    boiler (boiler_py.Boiler): The boiler instance supplying hot water for
      reheat.
    flow_rate_demand (float): Calculated air flow rate (m^3/s) demanded by this
      VAV based on its damper setting and max flow rate.
    reheat_demand (float): Calculated hot water flow rate (kg/s) demanded by
      this VAV for its reheat coil.
    zone_air_temperature (float): The current average air temperature (K) of the
      zone this VAV serves, as sensed by its thermostat.
  """

  def __init__(
      self,
      max_air_flow_rate_m3_s: float,
      reheat_max_water_flow_rate_kg_s: float,
      thermostat_instance: thermostat_py.Thermostat,
      boiler_instance: boiler_py.Boiler,
      device_id: Optional[str] = None,
      zone_id: Optional[str] = None,
  ):
    """Initializes the VAV unit.

    Args:
      max_air_flow_rate_m3_s (float): Maximum air flow rate (m^3/s) the VAV
        can deliver when its damper is fully open.
      reheat_max_water_flow_rate_kg_s (float): Maximum hot water flow rate
        (kg/s) through the reheat coil.
      thermostat_instance (thermostat_py.Thermostat): The thermostat that
        controls this VAV unit.
      boiler_instance (boiler_py.Boiler): The boiler that supplies hot water
        to this VAV's reheat coil.
      device_id (Optional[str]): A unique identifier for this VAV unit. If
        None, a UUID will be generated.
      zone_id (Optional[str]): The identifier of the zone this VAV serves. If
        None, a UUID-based zone ID will be generated.
    """
    observable_fields_info = {
        "supply_air_damper_percentage_command": smart_device.AttributeInfo(
            "damper_setting", float
        ),
        "supply_air_flowrate_setpoint": smart_device.AttributeInfo(
            "max_air_flow_rate", float # Represents VAV's capacity
        ),
        "zone_air_temperature_sensor": smart_device.AttributeInfo(
            "zone_air_temperature", float
        ),
    }
    action_fields_info = {
        # Damper setting is typically an action for an RL agent
        "supply_air_damper_percentage_command": smart_device.AttributeInfo(
            "damper_setting", float
        ),
        # Reheat valve setting could also be an action in some control schemes
    }

    dev_id = device_id if device_id else f"vav_id_{uuid.uuid4()}"
    zn_id = zone_id if zone_id else f"zone_id_for_vav_{dev_id}"

    super().__init__(
        observable_fields=observable_fields_info,
        action_fields=action_fields_info,
        device_type=smart_control_building_pb2.DeviceInfo.DeviceType.VAV,
        device_id=dev_id,
        zone_id=zn_id,
    )

    self._init_max_air_flow_rate_m3_s = max_air_flow_rate_m3_s
    self._init_reheat_max_water_flow_rate_kg_s = reheat_max_water_flow_rate_kg_s
    self._init_reheat_valve_setting = 0.0 # Start closed
    self._init_damper_setting = 0.1       # Start with minimum ventilation
    self._init_thermostat = thermostat_instance
    self._init_zone_air_temperature_k = 0.0 # Will be updated

    # Initialize state attributes (will be set in reset)
    self._max_air_flow_rate: float = 0.0
    self._reheat_max_water_flow_rate: float = 0.0
    self._reheat_valve_setting: float = 0.0
    self._damper_setting: float = 0.0
    self._thermostat: thermostat_py.Thermostat = thermostat_instance # Placeholder
    self._zone_air_temperature: float = 0.0
    self.reset() # Set initial state
    self._boiler: boiler_py.Boiler = boiler_instance


  def reset(self) -> None:
    """Resets the VAV unit to its initial state."""
    self._max_air_flow_rate = self._init_max_air_flow_rate_m3_s
    self._reheat_max_water_flow_rate = self._init_reheat_max_water_flow_rate_kg_s
    self._reheat_valve_setting = self._init_reheat_valve_setting
    self._damper_setting = self._init_damper_setting
    self._thermostat = self._init_thermostat # Re-assign thermostat instance
    self._zone_air_temperature = self._init_zone_air_temperature_k
    # Reset thermostat as well if it has state
    if hasattr(self._thermostat, "reset"):
        self._thermostat.reset() # type: ignore[attr-defined]
    logging.debug("VAV '%s' for zone '%s' reset.", self.device_id(), self.zone_id())

  @property
  def thermostat(self) -> thermostat_py.Thermostat:
    """thermostat_py.Thermostat: The thermostat controlling this VAV."""
    return self._thermostat

  @property
  def boiler(self) -> boiler_py.Boiler:
    """boiler_py.Boiler: The boiler supplying hot water for reheat."""
    return self._boiler

  @property
  def reheat_valve_setting(self) -> float:
    """Current setting of the reheat hot water valve (0.0 to 1.0)."""
    return self._reheat_valve_setting

  @reheat_valve_setting.setter
  def reheat_valve_setting(self, value: float) -> None:
    if not (0.0 <= value <= 1.0):
      raise ValueError(
          f"reheat_valve_setting must be in [0, 1], got {value}"
      )
    self._reheat_valve_setting = value

  @property
  def max_air_flow_rate(self) -> float:
    """Maximum air flow rate (m^3/s) capacity of this VAV."""
    return self._max_air_flow_rate

  @max_air_flow_rate.setter
  def max_air_flow_rate(self, value: float) -> None:
    if value < 0:
        raise ValueError("max_air_flow_rate cannot be negative.")
    self._max_air_flow_rate = value

  @property
  def damper_setting(self) -> float:
    """Current setting of the air damper (0.0 to 1.0)."""
    return self._damper_setting

  @damper_setting.setter
  def damper_setting(self, value: float) -> None:
    if not (0.0 <= value <= 1.0):
      raise ValueError(f"damper_setting must be in [0, 1], got {value}")
    self._damper_setting = value

  @property
  def flow_rate_demand(self) -> float:
    """Calculated air flow rate demand (m^3/s) based on damper and max flow."""
    return self._damper_setting * self._max_air_flow_rate

  @property
  def reheat_demand(self) -> float:
    """Calculated hot water flow rate demand (kg/s) for reheat."""
    return self._reheat_valve_setting * self._reheat_max_water_flow_rate

  @property
  def zone_air_temperature(self) -> float:
    """Current average air temperature (K) of the zone served by this VAV."""
    return self._zone_air_temperature

  def compute_reheat_energy_rate(
      self, supply_air_temp_k: float, boiler_supply_water_temp_k: float
  ) -> float:
    """Calculates thermal power (W) provided by the reheat coil.

    Args:
      supply_air_temp_k (float): Temperature (K) of air entering the VAV from
        the AHU (before reheat).
      boiler_supply_water_temp_k (float): Temperature (K) of hot water supplied
        by the boiler to the reheat coil.

    Returns:
      float: Thermal power (Watts) added by the reheat coil. Positive if
      water is hotter than air and valve is open.
    """
    actual_reheat_water_flow_kg_s = self.reheat_demand
    if actual_reheat_water_flow_kg_s == 0:
      return 0.0

    # Q_reheat = m_dot_water * C_p_water * (T_water_in - T_air_in)
    # This assumes perfect heat exchange efficiency for simplicity, where outlet
    # water temp would approach air temp, or air temp approaches water temp.
    # A more detailed model would use effectiveness-NTU or LMTD.
    # Here, it seems to calculate potential heat transfer from water to air.
    # If T_water_in <= T_air_in, no heat is transferred.
    delta_temp_k = max(0, boiler_supply_water_temp_k - supply_air_temp_k)
    reheat_power_watts = (
        actual_reheat_water_flow_kg_s *
        constants.WATER_SPECIFIC_HEAT_J_KGK *
        delta_temp_k
    )
    return reheat_power_watts

  def compute_zone_supply_temp(
      self, supply_air_temp_from_ahu_k: float, boiler_supply_water_temp_k: float
  ) -> float:
    """Calculates the final temperature of air supplied to the zone from VAV.

    This considers the effect of the reheat coil. If no air is flowing or no
    reheat is active, it returns the incoming supply air temperature.

    Args:
      supply_air_temp_from_ahu_k (float): Temperature (K) of air entering
        the VAV from the AHU.
      boiler_supply_water_temp_k (float): Temperature (K) of hot water from
        the boiler.

    Returns:
      float: Temperature (K) of air supplied to the zone after passing through
      the VAV (potentially reheated).
    """
    if self.damper_setting == 0 or self._max_air_flow_rate == 0:
      return supply_air_temp_from_ahu_k # No flow, no change

    reheat_power_watts = self.compute_reheat_energy_rate(
        supply_air_temp_from_ahu_k, boiler_supply_water_temp_k
    )
    if reheat_power_watts == 0:
      return supply_air_temp_from_ahu_k # No reheat, no change

    actual_air_flow_m3_s = self.flow_rate_demand
    air_mass_flow_kg_s = actual_air_flow_m3_s * constants.AIR_DENSITY

    if air_mass_flow_kg_s == 0: # Should not happen if damper > 0 and max_flow > 0
        return supply_air_temp_from_ahu_k

    # delta_T_air = Q_reheat / (m_dot_air * C_p_air)
    temp_increase_k = reheat_power_watts / (
        air_mass_flow_kg_s * constants.AIR_HEAT_CAPACITY
    )
    return supply_air_temp_from_ahu_k + temp_increase_k

  def compute_energy_applied_to_zone(
      self,
      current_zone_temp_k: float,
      supply_air_temp_from_ahu_k: float,
      boiler_supply_water_temp_k: float,
  ) -> float:
    """Calculates net thermal power (W) delivered to the zone by this VAV.

    This is the energy difference between the air supplied by the VAV (after
    reheat) and the current zone air, multiplied by the air mass flow rate.

    Args:
      current_zone_temp_k (float): Current average temperature (K) of the zone.
      supply_air_temp_from_ahu_k (float): Temperature (K) of air entering VAV.
      boiler_supply_water_temp_k (float): Temperature (K) of boiler hot water.

    Returns:
      float: Net thermal power (Watts) applied to the zone. Positive for
      heating the zone, negative for cooling.
    """
    if self.damper_setting == 0 or self._max_air_flow_rate == 0:
      return 0.0 # No airflow, no energy transfer

    final_supply_temp_k = self.compute_zone_supply_temp(
        supply_air_temp_from_ahu_k, boiler_supply_water_temp_k
    )
    actual_air_flow_m3_s = self.flow_rate_demand
    air_mass_flow_kg_s = actual_air_flow_m3_s * constants.AIR_DENSITY

    # Q_zone = m_dot_air * C_p_air * (T_supply_to_zone - T_zone)
    energy_to_zone_watts = (
        air_mass_flow_kg_s *
        constants.AIR_HEAT_CAPACITY *
        (final_supply_temp_k - current_zone_temp_k)
    )
    return energy_to_zone_watts

  def update_settings(
      self, current_zone_temp_k: float, current_sim_timestamp: pd.Timestamp
  ) -> None:
    """Updates VAV damper and reheat valve settings based on thermostat logic.

    Args:
      current_zone_temp_k (float): Current average temperature (K) of the zone.
      current_sim_timestamp (pd.Timestamp): Current simulation timestamp.
    """
    self._zone_air_temperature = current_zone_temp_k # Update sensed temperature
    thermostat_mode = self._thermostat.update(
        current_zone_temp_k, current_sim_timestamp
    )

    if thermostat_mode == thermostat_py.Thermostat.Mode.HEAT:
      self.damper_setting = 1.0       # Maximize airflow for heating
      self.reheat_valve_setting = 1.0 # Maximize reheat
    elif thermostat_mode == thermostat_py.Thermostat.Mode.COOL:
      self.damper_setting = 1.0       # Maximize airflow for cooling
      self.reheat_valve_setting = 0.0 # No reheat
    elif thermostat_mode == thermostat_py.Thermostat.Mode.OFF:
      self.damper_setting = 0.1       # Minimum ventilation
      self.reheat_valve_setting = 0.0
    elif thermostat_mode == thermostat_py.Thermostat.Mode.PASSIVE_COOL: # Or deadband
      self.damper_setting = 0.1       # Minimum ventilation (or could be adjusted)
      self.reheat_valve_setting = 0.0
    # else: implicit, keep existing settings if mode is not explicitly handled

  def output(
      self, current_zone_temp_k: float, supply_air_temp_from_ahu_k: float
  ) -> Tuple[float, float]:
    """Calculates VAV output: thermal power to zone and final supply air temp.

    This method assumes `update_settings` has already been called for the
    current state.

    Args:
      current_zone_temp_k (float): Current average temperature (K) of the zone.
      supply_air_temp_from_ahu_k (float): Temperature (K) of air entering VAV
        from the AHU.

    Returns:
      Tuple[float, float]:
        - q_to_zone_watts (float): Net thermal power (W) delivered to the zone.
        - final_vav_supply_temp_k (float): Temperature (K) of air supplied by
          this VAV to the zone after potential reheat.
    """
    self._zone_air_temperature = current_zone_temp_k # Ensure internal state is current
    boiler_supply_temp_k = self.boiler.supply_water_temperature_sensor

    q_to_zone_watts = self.compute_energy_applied_to_zone(
        current_zone_temp_k, supply_air_temp_from_ahu_k, boiler_supply_temp_k
    )
    final_vav_supply_temp_k = self.compute_zone_supply_temp(
        supply_air_temp_from_ahu_k, boiler_supply_temp_k
    )
    return q_to_zone_watts, final_vav_supply_temp_k

  def update(
      self,
      current_zone_temp_k: float,
      current_sim_timestamp: pd.Timestamp,
      supply_air_temp_from_ahu_k: float,
  ) -> Tuple[float, float]:
    """Updates VAV settings and then calculates its output.

    This is a convenience method combining `update_settings` and `output`.

    Args:
      current_zone_temp_k (float): Current average temperature (K) of the zone.
      current_sim_timestamp (pd.Timestamp): Current simulation timestamp.
      supply_air_temp_from_ahu_k (float): Temperature (K) of air supplied to
        this VAV from the AHU.

    Returns:
      Tuple[float, float]:
        - q_to_zone_watts (float): Net thermal power (W) delivered to the zone.
        - final_vav_supply_temp_k (float): Temperature (K) of air supplied by
          this VAV to the zone after potential reheat.
    """
    self.update_settings(current_zone_temp_k, current_sim_timestamp)
    return self.output(current_zone_temp_k, supply_air_temp_from_ahu_k)
