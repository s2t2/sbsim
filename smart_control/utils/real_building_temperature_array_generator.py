"""Generates a 2D temperature array from VAV sensor readings for visualization.

This module defines `RealBuildingTemperatureArrayGenerator`, a class that
takes temperature readings from an `ObservationResponse` (typically from VAV
zone temperature sensors) and maps them onto a 2D NumPy array representing the
building's floor plan. This temperature array can then be used for creating
thermal map visualizations.
"""

from typing import Mapping, Sequence, Tuple # Tuple was missing

import numpy as np
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils

# Type alias for a room/zone, represented as a sequence of (row, col) CV coordinates.
RoomCVCoordinates = Sequence[Tuple[int, int]]


class RealBuildingTemperatureArrayGenerator:
  """Generates a 2D temperature array from VAV temperature sensor readings.

  This class uses a building layout (floor plan), a mapping of device codes
  to their room/zone CV coordinates, and a mapping of device IDs to device codes
  to transform temperature data from an `ObservationResponse` into a 2D NumPy
  array. Each cell in the array corresponding to a Control Volume (CV) within
  a zone is filled with the temperature reported by the sensor in that zone.

  Attributes:
    _building_layout (np.ndarray): A 2D NumPy array representing the floor plan,
      used to determine the shape of the output temperature array.
    _device_code_to_room_cvs_map (Mapping[str, RoomCVCoordinates]): A dictionary
      mapping device codes (human-readable names) to a sequence of (row, col)
      tuples representing the CVs that constitute the room/zone associated
      with that device.
    _device_id_to_code_map (Mapping[str, str]): A dictionary mapping unique
      device IDs (e.g., UUIDs from the system) to their corresponding
      device codes.
  """

  def __init__(
      self,
      building_layout: np.ndarray,
      device_code_to_room_cvs_map: Mapping[str, RoomCVCoordinates],
      device_id_to_code_map: Mapping[str, str],
  ):
    """Initializes the RealBuildingTemperatureArrayGenerator.

    Args:
      building_layout (np.ndarray): A 2D NumPy array defining the shape and
        layout of the building (e.g., walls, interior spaces). The output
        temperature array will have the same shape.
      device_code_to_room_cvs_map (Mapping[str, RoomCVCoordinates]): A map
        where keys are device codes (e.g., "VAV_Zone1_TempSensor") and values
        are sequences of (row, col) tuples indicating the CVs covered by the
        zone associated with that device/sensor.
      device_id_to_code_map (Mapping[str, str]): A map from unique device IDs
        (as found in `ObservationResponse`) to device codes (used as keys in
        `device_code_to_room_cvs_map`).
    """
    self._building_layout: np.ndarray = building_layout
    self._device_code_to_room_cvs_map: Mapping[str, RoomCVCoordinates] = (
        device_code_to_room_cvs_map
    )
    self._device_id_to_code_map: Mapping[str, str] = device_id_to_code_map

  def get_temperature_array(
      self, observation_proto: smart_control_building_pb2.ObservationResponse
  ) -> Tuple[np.ndarray, pd.Timestamp]:
    """Generates a 2D temperature array and corresponding timestamp.

    The method processes an `ObservationResponse`, extracts temperature readings
    (assumed to be in Fahrenheit from "zone_air_temperature_sensor" fields),
    converts them to Kelvin, and populates a 2D NumPy array according to the
    building layout and device-to-room mappings.

    Args:
      observation_proto (smart_control_building_pb2.ObservationResponse):
        A protobuf message containing sensor readings, including zone air
        temperatures.

    Returns:
      Tuple[np.ndarray, pd.Timestamp]:
        - A 2D NumPy array representing the temperature (in Kelvin) at each
          Control Volume. CVs not associated with a sensor reading in the
          response will typically have a default value (e.g., 0.0 or NaN,
          depending on initialization of the array if not all CVs are covered).
        - A Pandas Timestamp representing the time of the observation,
          converted to UTC.
    """
    timestamp_utc = conversion_utils.proto_to_pandas_timestamp(
        observation_proto.timestamp
    )
    # Initialize temperature array (e.g., with zeros or NaNs)
    temperature_grid_k = np.zeros_like(self._building_layout, dtype=float)

    for single_response in observation_proto.single_observation_responses:
      request = single_response.single_observation_request
      if request.measurement_name != "zone_air_temperature_sensor":
        continue # Process only zone temperature sensors

      device_id = request.device_id
      device_code = self._device_id_to_code_map.get(device_id)
      if not device_code:
        continue # Device ID not recognized

      room_cvs = self._device_code_to_room_cvs_map.get(device_code)
      if not room_cvs:
        continue # Device code not mapped to any room CVs

      # Assuming continuous_value for temperature is in Fahrenheit as per original
      temp_fahrenheit = single_response.continuous_value
      temp_kelvin = conversion_utils.fahrenheit_to_kelvin(temp_fahrenheit)

      # Assign this temperature to all CVs belonging to this room/zone
      for r_coord, c_coord in room_cvs:
        if (0 <= r_coord < temperature_grid_k.shape[0] and
            0 <= c_coord < temperature_grid_k.shape[1]):
          temperature_grid_k[r_coord, c_coord] = temp_kelvin
        # Else: CV coordinate from mapping is out of bounds for the layout

    return temperature_grid_k, timestamp_utc
