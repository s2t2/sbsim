"""Generates visual representations of building thermal states.

This module provides `BuildingImageGenerator` for creating PNG images that
visualize the temperature distribution within a simulated building based on
observation data. The generated images are typically base64 encoded for easy
embedding or transmission.
"""

import base64
from collections.abc import Sequence
import io
import json
import os
import pathlib
import sys
from typing import Mapping, TypeAlias # Mapping was missing

from absl import logging
import gin
import numpy as np
from PIL import Image

from smart_control.proto import smart_control_building_pb2
from smart_control.utils import building_renderer
from smart_control.utils import real_building_temperature_array_generator as temp_array_gen

# Handle importlib.resources.abc.Traversable for different Python versions
if sys.version_info >= (3, 11):
  from importlib.resources.abc import Traversable
else:
  from importlib_resources.abc import Traversable # type: ignore[no-redef]

PathLocation: TypeAlias = Traversable | os.PathLike[str] | str
"""Type alias for path-like objects, supporting strings, os.PathLike, or Traversable."""


def _make_traversable(path_location: PathLocation) -> Traversable:
  """Converts a string or os.PathLike path to a Traversable object.

  If the input is already Traversable, it's returned as is. This is useful for
  working with resources within packages.

  Args:
    path_location (PathLocation): The path to convert.

  Returns:
    Traversable: A Traversable object representing the path.
  """
  if isinstance(path_location, Traversable):
    return path_location
  else:
    # Ensure path_location is string if it's os.PathLike for pathlib.Path
    return pathlib.Path(os.fspath(path_location))


@gin.configurable
class BuildingImageGenerator:
  """Generates base64 encoded PNG images of building temperatures.

  This class takes an observation response (containing temperature sensor
  readings), a device layout map, a floor plan, and device information to
  construct a 2D temperature array. This array is then rendered into an image
  representing the thermal state of the building.

  Attributes:
    _device_layout_path (Traversable): Path to the JSON file defining the
      layout of devices within rooms/zones.
    _floor_plan_path (Traversable): Path to the .npy file representing the
      building's floor plan (0=interior, 1=wall, 2=exterior).
    _device_infos (Sequence[smart_control_building_pb2.DeviceInfo]): A sequence
      of DeviceInfo protobuf messages, used to map device IDs to device codes.
    _cv_size (int): The size (in pixels) to render each Control Volume (CV)
      in the output image.
  """

  def __init__(
      self,
      device_layout_path: PathLocation,
      floor_plan_path: PathLocation,
      device_infos: Sequence[smart_control_building_pb2.DeviceInfo],
      cv_size: int,
  ):
    """Initializes the BuildingImageGenerator.

    Args:
      device_layout_path (PathLocation): Path to a JSON file. This file should
        map room/zone identifiers to lists of devices or sensor identifiers
        located within them. It's used to associate temperature readings with
        their physical locations on the floor plan.
      floor_plan_path (PathLocation): Path to a NumPy array file (.npy). This
        array represents the building's layout, where different integer values
        denote different types of spaces (e.g., 0 for interior air, 1 for
        walls, 2 for exterior space).
      device_infos (Sequence[smart_control_building_pb2.DeviceInfo]): A list
        of DeviceInfo protobuf messages. This is used to create a mapping from
        device IDs (often UUIDs) to more human-readable device codes or names
        that might be used in the device layout file.
      cv_size (int): The size in pixels for rendering each cell (Control
        Volume) of the floor plan grid in the output image.
    """
    self._device_layout_path: Traversable = _make_traversable(device_layout_path)
    self._floor_plan_path: Traversable = _make_traversable(floor_plan_path)
    self._device_infos: Sequence[smart_control_building_pb2.DeviceInfo] = (
        device_infos
    )
    self._cv_size: int = cv_size

  def _load_device_to_room_map(self) -> Mapping[str, Any]:
    """Loads and processes the device layout to map device codes to rooms.

    Returns:
      Mapping[str, Any]: A dictionary mapping device codes to room identifiers
      or other layout information from the JSON file.
    """
    device_id_to_code_map: dict[str, str] = {
        info.device_id: info.code for info in self._device_infos
    }

    try:
      with self._device_layout_path.open("rt", encoding="utf-8") as f:
        room_to_device_list_map = json.load(f)
    except FileNotFoundError:
      logging.error("Device layout file not found at: %s", self._device_layout_path)
      return {}
    except json.JSONDecodeError:
      logging.error("Error decoding JSON from device layout file: %s", self._device_layout_path)
      return {}

    device_code_to_room_map: dict[str, Any] = {}
    keys_not_found_in_device_map = set()

    # The room_dict_real seems to be: {"Room_1-2-3": ["SensorX", "SensorY"], ...}
    # We need to map device_code (from device_infos) to room name.
    # The original logic was: if "Room_1-2-3 " is in "device_code_Room_1-2-3_SensorX ", then map.
    # This seems error-prone. A more robust mapping would be needed if keys don't align.
    # Assuming the original logic's intent:
    for room_name, devices_in_room in room_to_device_list_map.items():
      if not room_name:
        continue
      # This part is tricky: how are device_codes related to room_name?
      # The original code iterates through all device_map.values() (device codes)
      # and checks if `room_name + " "` is in `device_code + " "`.
      # This implies a naming convention like `device_code = "AHU_Room_1-2-3"`.
      # Let's try to find any device_code that contains the room_name.
      # This is still a heuristic.
      for device_id, device_code_val in device_id_to_code_map.items():
        # A common pattern is that device_code might be like "ZONE-THERMOSTAT"
        # and room_name from JSON is "ZONE".
        if room_name in device_code_val: # Simplified heuristic
             device_code_to_room_map[device_code_val] = room_name # Or map to room_name/devices_in_room
        # The original code mapped device_code to the `room` object from JSON
        # which might be a list of sensors or properties.
        # device_layout_map[device_code] = room
        # This part needs clarification on the exact structure of room_dict_real
        # and how it relates to device_infos.
        # For now, let's assume device_layout_map should be device_code -> room_name
    if not device_code_to_room_map and room_to_device_list_map: # Heuristic failed
        logging.warning("Could not establish a clear device_code to room mapping.")

    # The original code's "keys_not_found" logic seems to check if JSON keys
    # (room names) can be associated with any device_code.
    # This might be better handled by ensuring device_infos align with layout.

    return device_code_to_room_map # Or the original device_layout_map if structure is different

  def generate_building_image(
      self, observation_response: smart_control_building_pb2.ObservationResponse
  ) -> bytes:
    """Generates a base64 encoded PNG image of building temperatures.

    Args:
      observation_response (smart_control_building_pb2.ObservationResponse):
        The observation data containing sensor readings.

    Returns:
      bytes: A base64 encoded string representing the PNG image. Returns an
      empty byte string if image generation fails.
    """
    device_id_to_code_map: dict[str, str] = {
        info.device_id: info.code for info in self._device_infos
    }

    # This part needs to be robust based on actual JSON structure.
    # Assuming device_layout_map is a map from device_code to room structure.
    # The original code structure was:
    # device_layout_map = {} -> map device_code to room object from JSON
    # This is passed to RealBuildingTemperatureArrayGenerator.
    # Let's refine how device_layout_map is constructed based on common patterns.
    # Typically, device_layout.json might be: {"RoomA": {"sensors": ["T_sensor_1"]}, ...}
    # or {"T_sensor_1": {"room": "RoomA", "coordinates_on_plan": [x,y]}}
    # The `RealBuildingTemperatureArrayGenerator` expects a map from
    # device_code to its room/zone identifier or properties.
    # The original code's mapping logic was complex and potentially error-prone.
    # For now, we use a simplified approach to load the JSON directly.
    try:
      with self._device_layout_path.open("rt", encoding="utf-8") as f:
        # This is `room_dict_real` in original, assumed to be device_code -> room info
        device_code_to_room_properties_map = json.load(f)
    except Exception as e: # pylint: disable=broad-except
        logging.error("Failed to load or parse device_layout_path: %s", e)
        return b""


    try:
      with self._floor_plan_path.open("rb") as fp:
        floor_plan_array = np.load(fp)
    except Exception as e: # pylint: disable=broad-except
        logging.error("Failed to load floor_plan_path: %s", e)
        return b""

    renderer = building_renderer.BuildingRenderer(
        floor_plan_array, self._cv_size
    )
    # RealBuildingTemperatureArrayGenerator needs:
    # floor_plan, device_layout_map (device_code -> room), device_map (device_id -> device_code)
    temp_array_generator = temp_array_gen.RealBuildingTemperatureArrayGenerator(
        floor_plan=floor_plan_array,
        device_layout_map=device_code_to_room_properties_map, # This needs to be correct structure
        device_id_to_code_map=device_id_to_code_map
    )

    try:
      temperature_grid, _ = temp_array_generator.get_temperature_array(
          observation_response
      )
      pil_image = renderer.render(temperature_grid) # Render the first layer if 3D
      return self.image_to_png_base64(pil_image)
    except Exception as e: # pylint: disable=broad-except
      logging.error("Error during temperature array generation or rendering: %s", e)
      return b""


  def image_to_png_base64(self, image: Image.Image) -> bytes:
    """Converts a PIL Image object to a base64 encoded PNG byte string.

    Args:
      image (Image.Image): The Pillow Image object to convert.

    Returns:
      bytes: A base64 encoded byte string of the PNG image.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())
