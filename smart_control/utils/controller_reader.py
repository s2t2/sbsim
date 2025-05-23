"""Utilities for reading Smart Control protobuf messages from files.

This module provides the `ProtoReader` class, which implements the
`BaseReader` interface for reading various Smart Control protobuf messages
(e.g., ObservationResponse, ActionResponse, RewardInfo, DeviceInfo) from
serialized files.

The reader assumes that data for time-series messages (like observations or
actions) is stored in hourly sharded files. Each file contains a stream of
protobuf messages, where each message is preceded by its size in bytes.
Static information like DeviceInfo or ZoneInfo is typically read from a single
file.
"""

import glob
import operator
import os
import re
from typing import Callable, Mapping, Sequence, TypeVar, Union

from absl import logging
import gin
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import constants
from smart_control.utils import reader_lib

# Generic type variable for protobuf messages.
ProtoMessageType = TypeVar("ProtoMessageType")


@gin.configurable
class ProtoReader(reader_lib.BaseReader):
  """Reads Smart Control protobuf messages from serialized files.

  This reader handles data that is potentially sharded by hour. Each shard file
  is expected to contain a sequence of protobuf messages, where each message is
  prefixed by a 4-byte little-endian integer indicating its size.

  File naming convention for sharded files is expected to include a timestamp
  suffix like `_YYYY.MM.DD.HH`. For example, observation responses for the 4th
  hour of May 25, 2021, might be in a file named
  `observation_response_2021.05.25.04.txtpb` (or similar, extension may vary).

  Attributes:
    _input_dir (str): The directory path where the protobuf data files are
      located.
  """

  def __init__(self, input_dir: str):
    """Initializes the ProtoReader.

    Args:
      input_dir (str): The directory path containing the serialized protobuf
        data files.
    """
    self._input_dir: str = input_dir
    logging.info("ProtoReader initialized for input directory: %s", self._input_dir)

  def read_observation_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Reads `ObservationResponse` messages within a time range.

    Args:
      start_time (pd.Timestamp): The inclusive start time of the query window.
      end_time (pd.Timestamp): The inclusive end time of the query window.

    Returns:
      Sequence[smart_control_building_pb2.ObservationResponse]: A list of
      `ObservationResponse` protobuf messages read from the files within the
      specified time range.
    """
    return self._read_messages(
        start_time,
        end_time,
        constants.OBSERVATION_RESPONSE_FILE_PREFIX,
        smart_control_building_pb2.ObservationResponse.FromString,
    )

  def read_action_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ActionResponse]:
    """Reads `ActionResponse` messages within a time range.

    Args:
      start_time (pd.Timestamp): The inclusive start time of the query window.
      end_time (pd.Timestamp): The inclusive end time of the query window.

    Returns:
      Sequence[smart_control_building_pb2.ActionResponse]: A list of
      `ActionResponse` protobuf messages.
    """
    return self._read_messages(
        start_time,
        end_time,
        constants.ACTION_RESPONSE_FILE_PREFIX,
        smart_control_building_pb2.ActionResponse.FromString,
    )

  def read_reward_infos(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads `RewardInfo` messages within a time range.

    Args:
      start_time (pd.Timestamp): The inclusive start time of the query window.
      end_time (pd.Timestamp): The inclusive end time of the query window.

    Returns:
      Sequence[smart_control_reward_pb2.RewardInfo]: A list of `RewardInfo`
      protobuf messages.
    """
    return self._read_messages(
        start_time,
        end_time,
        constants.REWARD_INFO_PREFIX,
        smart_control_reward_pb2.RewardInfo.FromString,
    )

  def read_reward_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardResponse]:
    """Reads `RewardResponse` messages within a time range.

    Args:
      start_time (pd.Timestamp): The inclusive start time of the query window.
      end_time (pd.Timestamp): The inclusive end time of the query window.

    Returns:
      Sequence[smart_control_reward_pb2.RewardResponse]: A list of
      `RewardResponse` protobuf messages.
    """
    return self._read_messages(
        start_time,
        end_time,
        constants.REWARD_RESPONSE_PREFIX,
        smart_control_reward_pb2.RewardResponse.FromString,
    )

  def read_zone_infos(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Reads `ZoneInfo` messages from a dedicated file.

    Assumes zone information is static and stored in a single file named
    according to `constants.ZONE_INFO_PREFIX`.

    Returns:
      Sequence[smart_control_building_pb2.ZoneInfo]: A list of `ZoneInfo`
      protobuf messages.
    """
    filename = os.path.join(self._input_dir, constants.ZONE_INFO_PREFIX)
    return self._read_streamed_protos(
        filename, smart_control_building_pb2.ZoneInfo.FromString
    )

  def read_device_infos(
      self,
  ) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Reads `DeviceInfo` messages from a dedicated file.

    Assumes device information is static and stored in a single file named
    according to `constants.DEVICE_INFO_PREFIX`.

    Returns:
      Sequence[smart_control_building_pb2.DeviceInfo]: A list of `DeviceInfo`
      protobuf messages.
    """
    filename = os.path.join(self._input_dir, constants.DEVICE_INFO_PREFIX)
    return self._read_streamed_protos(
        filename, smart_control_building_pb2.DeviceInfo.FromString
    )

  def _read_messages(
      self,
      start_time: pd.Timestamp,
      end_time: pd.Timestamp,
      file_prefix: str,
      from_string_func: Callable[
          [Union[bytearray, bytes, memoryview]], ProtoMessageType
      ],
  ) -> Sequence[ProtoMessageType]:
    """Reads all relevant proto messages from sharded files within a time range.

    Args:
      start_time (pd.Timestamp): Inclusive start time for filtering shards.
      end_time (pd.Timestamp): Inclusive end time for filtering shards.
      file_prefix (str): The prefix used to identify relevant shard files
        (e.g., "observation_response_").
      from_string_func (Callable): A function (e.g., `ProtoType.FromString`)
        that deserializes a byte string into a protobuf message object.

    Returns:
      Sequence[ProtoMessageType]: A list of deserialized protobuf messages of
      the specified type, read from the selected shards.
    """
    all_messages: list[ProtoMessageType] = []
    shard_filepaths = self._get_shards_in_range(
        start_time, end_time, file_prefix
    )

    for shard_path in shard_filepaths:
      try:
        messages_from_shard = self._read_streamed_protos(
            shard_path, from_string_func
        )
        all_messages.extend(messages_from_shard)
      except FileNotFoundError:
        logging.warning("Shard file not found: %s. Skipping.", shard_path)
      except Exception as e: # pylint: disable=broad-except
        logging.error("Error reading shard %s: %s", shard_path, e)
    return all_messages

  def _get_shards_in_range(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, file_prefix: str
  ) -> Sequence[str]:
    """Identifies shard files within the specified directory and time range.

    Args:
      start_time (pd.Timestamp): Inclusive start time.
      end_time (pd.Timestamp): Inclusive end time.
      file_prefix (str): Prefix for identifying relevant shard files.

    Returns:
      Sequence[str]: A list of full file paths for shards within the range.
    """
    matching_shards: list[str] = []
    try:
      for filename in os.listdir(self._input_dir):
        if filename.startswith(file_prefix):
          try:
            # Assumes timestamp is in 'YYYY.MM.DD.HH' format at end of filename
            # (before potential extension).
            match = re.search(r"(\d{4}\.\d{2}\.\d{2}\.\d{2})", filename)
            if match:
              shard_timestamp_str = match.group(1)
              # Timestamps in filenames are usually local; assume UTC if not specified.
              # Or, if they represent the *start* of the hour for that shard.
              shard_time = pd.Timestamp(shard_timestamp_str, tz="UTC") # Or appropriate tz
              # Shard represents data for the hour *starting* at shard_time.
              # Include if [shard_time, shard_time + 1hr) overlaps with [start_time, end_time].
              shard_end_time = shard_time + pd.Timedelta(hours=1)
              if shard_time < end_time and shard_end_time > start_time:
                matching_shards.append(os.path.join(self._input_dir, filename))
          except (IndexError, ValueError) as e:
            logging.warning(
                "Could not parse timestamp from filename %s: %s", filename, e
            )
    except FileNotFoundError:
      logging.error("Input directory not found: %s", self._input_dir)
      return []
    return sorted(matching_shards) # Sort for chronological processing

  def _read_streamed_protos(
      self,
      full_filepath: str,
      from_string_func: Callable[
          [Union[bytearray, bytes, memoryview]], ProtoMessageType
      ],
  ) -> Sequence[ProtoMessageType]:
    """Reads a sequence of protos from a file where each is size-prefixed.

    Each protobuf message in the file is expected to be preceded by a 4-byte
    little-endian integer indicating the size of the serialized message.

    Args:
      full_filepath (str): The full path to the file to read.
      from_string_func (Callable): Function to deserialize bytes to a proto message.

    Returns:
      Sequence[ProtoMessageType]: A list of deserialized protobuf messages.
    """
    messages: list[ProtoMessageType] = []
    try:
      with open(full_filepath, "rb") as f:
        while True:
          size_bytes = f.read(4)
          if not size_bytes: # End of file
            break
          message_size = int.from_bytes(size_bytes, byteorder="little")
          serialized_data = f.read(message_size)
          if len(serialized_data) < message_size:
            logging.warning(
                "Premature EOF in %s: expected %d bytes, got %d.",
                full_filepath, message_size, len(serialized_data)
            )
            break
          messages.append(from_string_func(serialized_data))
    except FileNotFoundError:
      logging.error("File not found for reading streamed protos: %s", full_filepath)
      return [] # Return empty list if file doesn't exist
    except Exception as e: # pylint: disable=broad-except
      logging.error("Error reading streamed protos from %s: %s", full_filepath, e)
      return [] # Return any successfully read messages before error
    return messages

  def read_normalization_info(
      self,
  ) -> Mapping[
      reader_lib.VariableId,
      smart_control_normalization_pb2.ContinuousVariableInfo,
  ]:
    """Reads normalization parameters for continuous variables.

    The data is read from a file specified by
    `constants.NORMALIZATION_FILENAME` within the `_input_dir`. Each entry
    is a `ContinuousVariableInfo` protobuf message, size-prefixed.

    Returns:
      Mapping[reader_lib.VariableId, ContinuousVariableInfo]: A dictionary
      mapping variable IDs to their normalization information.

    Raises:
      ValueError: If a duplicate variable ID is found in the normalization file.
    """
    filepath = os.path.join(self._input_dir, constants.NORMALIZATION_FILENAME)
    normalization_info_map: dict[
        reader_lib.VariableId,
        smart_control_normalization_pb2.ContinuousVariableInfo,
    ] = {}
    try:
      with open(filepath, "rb") as f:
        while True:
          size_bytes = f.read(4)
          if not size_bytes:
            break
          message_size = int.from_bytes(size_bytes, byteorder="little")
          serialized_data = f.read(message_size)
          if len(serialized_data) < message_size:
            logging.warning("Premature EOF in normalization file %s.", filepath)
            break
          variable_info = smart_control_normalization_pb2.ContinuousVariableInfo()
          variable_info.FromString(serialized_data)
          variable_id = reader_lib.VariableId(variable_info.id)
          if variable_id in normalization_info_map:
            raise ValueError(
                f"Duplicate entry for variable ID '{variable_id}' found in "
                f"normalization file: {filepath}"
            )
          normalization_info_map[variable_id] = variable_info
    except FileNotFoundError:
      logging.error("Normalization info file not found: %s", filepath)
    except Exception as e: # pylint: disable=broad-except
      logging.error("Error reading normalization info from %s: %s", filepath, e)
    return normalization_info_map


def get_episode_data(working_dir: str) -> pd.DataFrame:
  """Extracts summary information about experiment episodes from a directory.

  This function scans a `working_dir` for subdirectories, each assumed to
  represent an episode of an experiment run. It parses timestamps from
  directory names and internal observation files to create a summary DataFrame.

  Directory/file naming conventions assumed:
  - Episode directory: `[label]_[yymmdd_hhmmss_UTC]`
    (e.g., `sac_collect_230115_103000`)
  - Observation files within episode directory: `[prefix]_[YYYY.MM.DD.HH_local].*`
    (e.g., `observation_response_2023.01.15.10.txtpb`)

  Args:
    working_dir (str): The path to the directory containing subdirectories for
      each experiment episode.

  Returns:
    pd.DataFrame: A DataFrame summarizing each episode, with columns for
    execution time (UTC from dir name), episode start/end times (local from
    obs files), duration, number of updates (observation files), and a label
    extracted from the directory name. Sorted by execution time.
  """
  try:
    episode_dir_names = os.listdir(working_dir)
  except FileNotFoundError:
    logging.error("Working directory for episode data not found: %s", working_dir)
    return pd.DataFrame() # Return empty DataFrame

  # Regex to extract timestamp from directory name (yymmdd_hhmmss)
  dir_ts_pattern = re.compile(r"_(\d{6}_\d{6})$")
  # Regex to extract timestamp from observation filename (YYYY.MM.DD.HH)
  obs_file_ts_pattern = re.compile(r"_(\d{4}\.\d{2}\.\d{2}\.\d{2})")

  episode_summaries = []

  for dir_name in episode_dir_names:
    dir_ts_match = dir_ts_pattern.search(dir_name)
    if not dir_ts_match:
      logging.warning("Could not parse execution timestamp from dir: %s", dir_name)
      continue
    try:
      exec_time = pd.to_datetime(dir_ts_match.group(1), format="%y%m%d_%H%M%S", utc=True)
      label = dir_name[:dir_ts_match.start()] # Part before the timestamp
    except ValueError:
      logging.warning("Error parsing execution timestamp from dir: %s", dir_name)
      continue

    episode_path = os.path.join(working_dir, dir_name)
    if not os.path.isdir(episode_path):
      continue

    obs_files = glob.glob(os.path.join(episode_path, f"{constants.OBSERVATION_RESPONSE_FILE_PREFIX}*"))
    obs_file_timestamps = []
    for obs_file in obs_files:
      obs_ts_match = obs_file_ts_pattern.search(os.path.basename(obs_file))
      if obs_ts_match:
        try:
          # Assume filenames use local time for the building/simulation
          obs_file_timestamps.append(
              pd.to_datetime(obs_ts_match.group(1), format="%Y.%m.%d.%H")
          )
        except ValueError:
          logging.warning("Error parsing obs file timestamp: %s", obs_file)

    if obs_file_timestamps:
      # Assuming timestamps in files are local; make them timezone-aware if needed.
      # For now, treat as naive then potentially localize if a default tz is known.
      # If constants.DEFAULT_TIME_ZONE is available and relevant:
      # obs_pd_timestamps = [t.tz_localize(constants.DEFAULT_TIME_ZONE) for t in obs_file_timestamps]
      obs_pd_timestamps = pd.Series(obs_file_timestamps) # Keep as naive for now
      start_time = obs_pd_timestamps.min()
      end_time = obs_pd_timestamps.max()
      duration_sec = (end_time - start_time).total_seconds()
      episode_summaries.append({
          "execution_time_utc": exec_time,
          "episode_start_time_local": start_time,
          "episode_end_time_local": end_time,
          "duration_seconds": duration_sec,
          "num_observation_files": len(obs_files),
          "label": label,
          "episode_directory": dir_name,
      })

  if not episode_summaries:
    return pd.DataFrame()

  episodes_df = pd.DataFrame(episode_summaries)
  return episodes_df.sort_values(by="execution_time_utc").set_index(
      "episode_directory"
  )
