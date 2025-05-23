"""Utilities for writing Smart Control protobuf messages to files.

This module provides the `ProtoWriter` class, which implements the
`BaseWriter` interface for serializing and writing various Smart Control
protobuf messages (e.g., ObservationResponse, ActionResponse, RewardInfo)
to disk.

The writer typically stores time-series data in hourly sharded files. Each
message within a file is prefixed by its size in bytes to facilitate streaming
reads. Static information like DeviceInfo or ZoneInfo is usually written to
a single, dedicated file, potentially overwriting it if it already exists.
"""

import csv
import os
from typing import Mapping, Sequence

from absl import logging
import gin
from google.protobuf import message # For type hinting proto messages
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import constants
from smart_control.utils import writer_lib # Contains BaseWriter and PathLocation


@gin.configurable
class ProtoWriter(writer_lib.BaseWriter):
  """Writes Smart Control protobuf messages to disk, often in sharded files.

  This class handles the serialization of various protobuf messages defined in
  the Smart Control project and writes them to files. For time-series data
  like observations, actions, and rewards, it typically creates hourly shards.
  Each message in these files is written with a 4-byte little-endian prefix
  indicating its size.

  Static data like device and zone information is written to specific,
  non-sharded files.

  Attributes:
    _output_dir (str): The base directory where all output files will be
      written.
  """

  def __init__(self, output_dir: str):
    """Initializes the ProtoWriter with a specific output directory.

    Args:
      output_dir (str): The path to the directory where data files will be
        created. The directory will be created if it does not exist.
    """
    self._output_dir: str = output_dir
    os.makedirs(self._output_dir, exist_ok=True)
    logging.info("ProtoWriter output directory set to: %s", self._output_dir)

  def _get_serial_from_timestamp(self, timestamp: pd.Timestamp) -> str:
    """Generates a string serial (YYYY.MM.DD.HH) from a pandas Timestamp.

    Args:
      timestamp (pd.Timestamp): The timestamp to convert.

    Returns:
      str: The formatted serial string.
    """
    return timestamp.strftime("%Y.%m.%d.%H")

  def _get_sharded_filepath(
      self, file_prefix: str, timestamp_serial: str
  ) -> str:
    """Constructs the full filepath for a sharded data file.

    Args:
      file_prefix (str): The prefix for the filename (e.g.,
        `constants.OBSERVATION_RESPONSE_FILE_PREFIX`).
      timestamp_serial (str): The hourly serial string (YYYY.MM.DD.HH) for
        the shard.

    Returns:
      str: The absolute path to the shard file.
    """
    return os.path.join(self._output_dir, f"{file_prefix}_{timestamp_serial}")

  def _write_proto_message_to_file(
      self, proto_message: message.Message, filepath: str
  ) -> None:
    """Writes a single protobuf message to a file with a size prefix.

    The file is opened in append binary mode ('ab') if it exists, otherwise
    in write binary mode ('wb'). Each message is preceded by a 4-byte
    little-endian integer representing its size.

    Args:
      proto_message (message.Message): The protobuf message to serialize and write.
      filepath (str): The full path to the file.
    """
    mode = "ab" if os.path.exists(filepath) else "wb"
    try:
      with open(filepath, mode) as output_f:
        serialized_msg = proto_message.SerializeToString()
        size_bytes = len(serialized_msg).to_bytes(4, byteorder="little")
        output_f.write(size_bytes)
        output_f.write(serialized_msg)
    except IOError as e:
      logging.error(
          "IOException encountered while writing proto to %s: %s", filepath, e
      )
    except Exception as e: # pylint: disable=broad-except
      logging.error("Unexpected error writing proto to %s: %s", filepath, e)


  def write_observation_response(
      self,
      observation_response: smart_control_building_pb2.ObservationResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes an `ObservationResponse` to an hourly sharded file.

    Args:
      observation_response (smart_control_building_pb2.ObservationResponse):
        The observation response protobuf message.
      timestamp (pd.Timestamp): The timestamp associated with this observation,
        used to determine the correct hourly shard.
    """
    serial = self._get_serial_from_timestamp(timestamp)
    filepath = self._get_sharded_filepath(
        constants.OBSERVATION_RESPONSE_FILE_PREFIX, serial
    )
    self._write_proto_message_to_file(observation_response, filepath)

  def write_building_image(
      self, base64_encoded_image: bytes, timestamp: pd.Timestamp
  ) -> None:
    """Appends a timestamp and base64 encoded image to a CSV file.

    Args:
      base64_encoded_image (bytes): The base64 encoded image data.
      timestamp (pd.Timestamp): The timestamp associated with the image.
    """
    filepath = os.path.join(self._output_dir, constants.BUILDING_IMAGE_CSV_FILE)
    # Ensure base64_encoded_image is a string for CSV writing
    image_str = base64_encoded_image.decode('utf-8') if isinstance(base64_encoded_image, bytes) else base64_encoded_image
    try:
      with open(filepath, "a", encoding="utf-8", newline="") as csv_f:
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow([timestamp.timestamp(), image_str])
    except IOError as e:
      logging.error("IOException writing building image to CSV %s: %s", filepath, e)


  def write_action_response(
      self,
      action_response: smart_control_building_pb2.ActionResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes an `ActionResponse` to an hourly sharded file.

    Args:
      action_response (smart_control_building_pb2.ActionResponse): The action
        response protobuf message.
      timestamp (pd.Timestamp): The timestamp for sharding.
    """
    serial = self._get_serial_from_timestamp(timestamp)
    filepath = self._get_sharded_filepath(
        constants.ACTION_RESPONSE_FILE_PREFIX, serial
    )
    self._write_proto_message_to_file(action_response, filepath)

  def write_reward_info(
      self,
      reward_info: smart_control_reward_pb2.RewardInfo,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes a `RewardInfo` message to an hourly sharded file.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): The reward info
        protobuf message.
      timestamp (pd.Timestamp): The timestamp for sharding.
    """
    serial = self._get_serial_from_timestamp(timestamp)
    filepath = self._get_sharded_filepath(
        constants.REWARD_INFO_PREFIX, serial
    )
    self._write_proto_message_to_file(reward_info, filepath)

  def write_reward_response(
      self,
      reward_response: smart_control_reward_pb2.RewardResponse,
      timestamp: pd.Timestamp,
  ) -> None:
    """Writes a `RewardResponse` message to an hourly sharded file.

    Args:
      reward_response (smart_control_reward_pb2.RewardResponse): The reward
        response protobuf message.
      timestamp (pd.Timestamp): The timestamp for sharding.
    """
    serial = self._get_serial_from_timestamp(timestamp)
    filepath = self._get_sharded_filepath(
        constants.REWARD_RESPONSE_PREFIX, serial
    )
    self._write_proto_message_to_file(reward_response, filepath)

  def write_normalization_info(
      self,
      normalization_info_map: Mapping[
          writer_lib.VariableId,
          smart_control_normalization_pb2.ContinuousVariableInfo,
      ],
  ) -> None:
    """Writes normalization parameters for continuous variables to a file.

    Each `ContinuousVariableInfo` message is size-prefixed. The file is
    overwritten if it already exists.

    Args:
      normalization_info_map (Mapping[VariableId, ContinuousVariableInfo]):
        A dictionary mapping variable IDs to their normalization information.
    """
    filepath = os.path.join(self._output_dir, constants.NORMALIZATION_FILENAME)
    try:
      # Overwrite the file each time this is called for static info.
      with open(filepath, "wb") as output_f:
        for variable_info in normalization_info_map.values():
          serialized_msg = variable_info.SerializeToString()
          size_bytes = len(serialized_msg).to_bytes(4, byteorder="little")
          output_f.write(size_bytes)
          output_f.write(serialized_msg)
    except IOError as e:
      logging.error("IOException writing normalization info to %s: %s", filepath, e)


  def write_device_infos(
      self, device_infos: Sequence[smart_control_building_pb2.DeviceInfo]
  ) -> None:
    """Writes a sequence of `DeviceInfo` messages to a dedicated file.

    The file is overwritten if it already exists. Each message is size-prefixed.

    Args:
      device_infos (Sequence[smart_control_building_pb2.DeviceInfo]): A list
        of `DeviceInfo` protobuf messages.
    """
    filepath = os.path.join(self._output_dir, constants.DEVICE_INFO_PREFIX)
    if os.path.exists(filepath):
      logging.info("Overwriting existing DeviceInfo file: %s", filepath)
      try:
        os.remove(filepath)
      except OSError as e:
        logging.error("Error deleting existing DeviceInfo file %s: %s", filepath, e)
        return # Abort if cannot remove existing file

    for device_info in device_infos:
      self._write_proto_message_to_file(device_info, filepath)

  def write_zone_infos(
      self, zone_infos: Sequence[smart_control_building_pb2.ZoneInfo]
  ) -> None:
    """Writes a sequence of `ZoneInfo` messages to a dedicated file.

    The file is overwritten if it already exists. Each message is size-prefixed.

    Args:
      zone_infos (Sequence[smart_control_building_pb2.ZoneInfo]): A list of
        `ZoneInfo` protobuf messages.
    """
    filepath = os.path.join(self._output_dir, constants.ZONE_INFO_PREFIX)
    if os.path.exists(filepath):
      logging.info("Overwriting existing ZoneInfo file: %s", filepath)
      try:
        os.remove(filepath)
      except OSError as e:
        logging.error("Error deleting existing ZoneInfo file %s: %s", filepath, e)
        return

    for zone_info in zone_infos:
      self._write_proto_message_to_file(zone_info, filepath)


@gin.configurable
class ProtoWriterFactory(writer_lib.BaseWriterFactory):
  """A factory for creating `ProtoWriter` instances.

  This class allows for the creation of `ProtoWriter` objects, typically
  configured via Gin, by specifying the output directory.
  """

  def create(self, output_dir: writer_lib.PathLocation) -> ProtoWriter:
    """Creates a `ProtoWriter` instance for the specified output directory.

    Args:
      output_dir (writer_lib.PathLocation): The directory path where the
        `ProtoWriter` will save its files. Can be a string, os.PathLike, or
        importlib.resources.abc.Traversable.

    Returns:
      ProtoWriter: An initialized `ProtoWriter` instance.
    """
    # Ensure output_dir is a string path for ProtoWriter constructor
    str_output_dir = os.fspath(output_dir)
    return ProtoWriter(str_output_dir)
