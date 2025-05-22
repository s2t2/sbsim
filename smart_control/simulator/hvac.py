"""Simulates a Heating, Ventilation, and Air Conditioning (HVAC) system.

This module provides the `Hvac` class, which models a centralized HVAC system
typically found in commercial buildings. The model generally assumes a primary
air handling unit (AHU) and a boiler providing conditioned air and hot water,
respectively, to multiple zones. Each zone is equipped with a Variable Air
Volume (VAV) unit and controlled by a thermostat operating on a predefined
schedule.
"""

from typing import List, Mapping, Tuple, Dict # Added Dict for type hint

import gin
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.simulator import air_handler as air_handler_py
from smart_control.simulator import boiler as boiler_py
from smart_control.simulator import setpoint_schedule
from smart_control.simulator import thermostat
from smart_control.simulator import vav
from smart_control.utils import conversion_utils


@gin.configurable
class Hvac:
  """Manages and coordinates the components of a simulated HVAC system.

  This class acts as an aggregator for the main HVAC components: a central
  air handler, a boiler, and a collection of Variable Air Volume (VAV) units,
  where each VAV serves a specific zone. It initializes these components and
  provides access to them, as well as managing metadata about the zones.

  The behavior of thermostats associated with each VAV is governed by a shared
  setpoint schedule.
  """

  def __init__(
      self,
      zone_coordinates: List[Tuple[int, int]],
      air_handler: air_handler_py.AirHandler,
      boiler: boiler_py.Boiler,
      schedule: setpoint_schedule.SetpointSchedule,
      vav_max_air_flow_rate: float,
      vav_reheat_max_water_flow_rate: float,
  ):
    """Initializes the HVAC system model.

    Args:
      zone_coordinates: A list of 2-tuples `(row, col)`, where each tuple
        uniquely identifies a zone that will be served by a dedicated VAV unit.
        These coordinates might correspond to a grid representation of the building.
      air_handler: An initialized instance of `air_handler_py.AirHandler` that
        will serve as the central air conditioning and ventilation unit.
      boiler: An initialized instance of `boiler_py.Boiler` that will provide
        hot water, typically for VAV reheat coils.
      schedule: An instance of `setpoint_schedule.SetpointSchedule` which defines
        the temperature setpoints (e.g., heating and cooling) over time. This
        schedule is used by the thermostats controlling each VAV unit.
      vav_max_air_flow_rate: The maximum air flow rate (in cubic meters per
        second, m^3/s) that each VAV unit can deliver.
      vav_reheat_max_water_flow_rate: The maximum hot water flow rate (in cubic
        meters per second, m^3/s, assuming consistent volumetric flow units)
        for the reheat coil within each VAV unit.
    """
    self._air_handler: air_handler_py.AirHandler = air_handler
    self._boiler: boiler_py.Boiler = boiler
    self._vav_max_air_flow_rate: float = vav_max_air_flow_rate
    self._vav_reheat_max_water_flow_rate: float = vav_reheat_max_water_flow_rate
    self._zone_coordinates: List[Tuple[int, int]] = zone_coordinates
    self._vavs: Dict[Tuple[int, int], vav.Vav] = {}
    self._schedule: setpoint_schedule.SetpointSchedule = schedule
    self._zone_infos: Dict[Tuple[int, int], smart_control_building_pb2.ZoneInfo] = {}

    # Create a VAV, Thermostat, and ZoneInfo for each specified zone coordinate
    for z_coord in self._zone_coordinates:
      zone_id_str = conversion_utils.zone_coordinates_to_id(z_coord)
      # Each VAV gets its own thermostat, but all thermostats use the same schedule
      vav_thermostat = thermostat.Thermostat(self._schedule)
      vav_device_id = f'vav_{z_coord[0]}_{z_coord[1]}'

      self._vavs[z_coord] = vav.Vav(
          max_air_flow_rate=self._vav_max_air_flow_rate,
          reheat_max_water_flow_rate=self._vav_reheat_max_water_flow_rate,
          thermostat=vav_thermostat,
          boiler=self._boiler, # VAVs are connected to the central boiler
          device_id=vav_device_id,
          zone_id=zone_id_str,
      )
      self._zone_infos[z_coord] = smart_control_building_pb2.ZoneInfo(
          zone_id=zone_id_str,
          building_id='US-SIM-001', # Example building ID
          zone_description=f'Simulated zone at coordinates {z_coord}',
          devices=[vav_device_id], # List of devices serving this zone
          zone_type=smart_control_building_pb2.ZoneInfo.ZoneType.ROOM, # Example type
          floor=0, # Example floor
      )
    self.reset() # Initialize states of all components

  def reset(self) -> None:
    """Resets all components of the HVAC system to their initial states.

    This involves calling the `reset()` method on the central air handler,
    the boiler, and each individual VAV unit.
    """
    self.air_handler.reset()
    self.boiler.reset()
    for z_coord in self._zone_coordinates:
      if z_coord in self._vavs: # Ensure VAV exists for the coordinate
        self._vavs[z_coord].reset()

  @property
  def vavs(self) -> Mapping[Tuple[int, int], vav.Vav]:
    """A mapping from zone coordinates to their respective `Vav` instances."""
    return self._vavs

  @property
  def air_handler(self) -> air_handler_py.AirHandler:
    """The central `AirHandler` instance for this HVAC system."""
    return self._air_handler

  @property
  def boiler(self) -> boiler_py.Boiler:
    """The central `Boiler` instance for this HVAC system."""
    return self._boiler

  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Checks if the HVAC system is currently in a "comfort" mode.

    This determination is delegated to the `SetpointSchedule` instance,
    which typically defines periods of active climate control based on time
    (e.g., occupied hours).

    Args:
      current_time: A `pandas.Timestamp` representing the current time for
        which to check the comfort mode.

    Returns:
      True if the system is in comfort mode at `current_time`, False otherwise.
    """
    return self._schedule.is_comfort_mode(current_time)

  @property
  def zone_infos(
      self,
  ) -> Mapping[Tuple[int, int], smart_control_building_pb2.ZoneInfo]:
    """A mapping from zone coordinates to `ZoneInfo` protobuf messages.
    
    Each `ZoneInfo` message contains metadata about a specific zone, such as
    its ID, description, and associated devices.
    """
    return self._zone_infos
