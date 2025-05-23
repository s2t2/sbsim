"""Models the Heating, Ventilation, and Air Conditioning (HVAC) system.

This module defines the `Hvac` class, which represents the collective HVAC
system for a simulated building. It typically includes central components like
an air handler and a boiler, and distributed components like Variable Air
Volume (VAV) units, one for each thermal zone in the building.

The HVAC model coordinates the operation of these components based on thermostat
setpoints, schedules, and control signals (potentially from an RL agent).
"""

from typing import List, Mapping, Tuple

import gin
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import thermostat as thermostat_py # Renamed to avoid conflict
from smart_control.simulator import vav as vav_py # Renamed to avoid conflict
from smart_control.utils import conversion_utils


@gin.configurable
class Hvac:
  """Represents the HVAC system of a simulated building.

  This class aggregates and manages the core HVAC components: a central air
  handler, a central boiler, and a set of Variable Air Volume (VAV) units,
  typically one per zone. Each VAV unit is associated with a thermostat that
  follows a defined setpoint schedule.

  The `Hvac` class provides access to these components and can determine if the
  building is operating in "comfort mode" based on the schedule.

  Attributes:
    vavs (Mapping[Tuple[int, int], vav_py.Vav]): A mapping from zone
      coordinates (row, col) to their corresponding VAV unit instance.
    air_handler (air_handler_py.AirHandler): The central air handler unit.
    boiler (boiler_py.Boiler): The central boiler unit.
    zone_infos (Mapping[Tuple[int, int], smart_control_building_pb2.ZoneInfo]):
      A mapping from zone coordinates to their `ZoneInfo` protobuf messages,
      containing metadata about each zone.
  """

  def __init__(
      self,
      zone_coordinates_list: List[Tuple[int, int]],
      air_handler_instance: air_handler_py.AirHandler,
      boiler_instance: boiler_py.Boiler,
      thermostat_setpoint_schedule: setpoint_schedule.SetpointSchedule,
      vav_max_air_flow_rate_m3_s: float,
      vav_reheat_max_water_flow_rate_kg_s: float,
  ):
    """Initializes the HVAC system model.

    Args:
      zone_coordinates_list (List[Tuple[int, int]]): A list of (row, col)
        tuples, where each tuple represents the coordinates of a thermal zone
        that will be serviced by a VAV unit.
      air_handler_instance (air_handler_py.AirHandler): An instance of the
        `AirHandler` class.
      boiler_instance (boiler_py.Boiler): An instance of the `Boiler` class.
      thermostat_setpoint_schedule (setpoint_schedule.SetpointSchedule): The
        setpoint schedule to be used by all thermostats controlling the VAVs.
      vav_max_air_flow_rate_m3_s (float): The maximum air flow rate (m^3/s)
        for each VAV unit.
      vav_reheat_max_water_flow_rate_kg_s (float): The maximum hot water flow
        rate (kg/s) for the reheat coil in each VAV unit.
    """
    self._air_handler: air_handler_py.AirHandler = air_handler_instance
    self._boiler: boiler_py.Boiler = boiler_instance
    self._vav_max_air_flow_rate_m3_s = vav_max_air_flow_rate_m3_s
    self._vav_reheat_max_water_flow_rate_kg_s = (
        vav_reheat_max_water_flow_rate_kg_s
    )
    self._zone_coordinates: List[Tuple[int, int]] = zone_coordinates_list
    self._schedule: setpoint_schedule.SetpointSchedule = (
        thermostat_setpoint_schedule
    )

    self._vavs: dict[Tuple[int, int], vav_py.Vav] = {}
    self._zone_infos: dict[
        Tuple[int, int], smart_control_building_pb2.ZoneInfo
    ] = {}

    for coords in self._zone_coordinates:
      zone_id_str = conversion_utils.zone_coordinates_to_id(coords)
      # Each VAV gets its own thermostat instance, but they share the schedule.
      vav_thermostat = thermostat_py.Thermostat(self._schedule)
      vav_device_id = f"vav_{coords[0]}_{coords[1]}"

      self._vavs[coords] = vav_py.Vav(
          max_air_flow_rate_m3_s=self._vav_max_air_flow_rate_m3_s,
          max_reheat_water_flow_rate_kg_s=(
              self._vav_reheat_max_water_flow_rate_kg_s
          ),
          thermostat_instance=vav_thermostat,
          boiler_instance=self._boiler, # VAVs are connected to the central boiler
          device_id=vav_device_id,
          zone_id=zone_id_str,
      )
      # Store metadata about the zone
      self._zone_infos[coords] = smart_control_building_pb2.ZoneInfo(
          zone_id=zone_id_str,
          building_id="US-SIM-001", # Example building ID
          zone_description=f"Simulated zone at coordinates {coords}",
          devices=[vav_device_id], # List devices in this zone
          zone_type=smart_control_building_pb2.ZoneInfo.ZoneType.ROOM,
          floor=0, # Example floor number
      )
    self.reset()

  def reset(self) -> None:
    """Resets all HVAC components to their initial states."""
    self.air_handler.reset()
    self.boiler.reset()
    for zone_coords_tuple in self._zone_coordinates:
      self._vavs[zone_coords_tuple].reset()
    logging.info("HVAC system reset.")


  @property
  def vavs(self) -> Mapping[Tuple[int, int], vav_py.Vav]:
    """Mapping[Tuple[int, int], vav_py.Vav]: VAV units by zone coordinates."""
    return self._vavs

  @property
  def air_handler(self) -> air_handler_py.AirHandler:
    """air_handler_py.AirHandler: The central air handler unit."""
    return self._air_handler

  @property
  def boiler(self) -> boiler_py.Boiler:
    """boiler_py.Boiler: The central boiler unit."""
    return self._boiler

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Checks if the building is scheduled to be in comfort mode.

    This is determined by the underlying setpoint schedule shared by
    thermostats.

    Args:
      current_time (pd.Timestamp): The current simulation time.

    Returns:
      bool: True if the schedule indicates comfort mode at `current_time`,
      False otherwise.
    """
    return self._schedule.is_comfort_mode(current_time)

  @property
  def zone_infos(
      self,
  ) -> Mapping[Tuple[int, int], smart_control_building_pb2.ZoneInfo]:
    """Mapping from zone coordinates to `ZoneInfo` protobuf messages."""
    return self._zone_infos
