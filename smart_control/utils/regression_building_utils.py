"""Utility functions for processing data related to regression-based building models.

This module provides helper functions for:
- Generating and expanding time-based features (e.g., hour of day, day of week)
  into cyclical sine/cosine representations.
- Converting protobuf messages (ObservationResponse, ActionResponse, RewardInfo)
  into Pandas DataFrames or Python dictionaries suitable for model input/output.
- Extracting specific information (like device action tuples or feature tuples)
  from these protobuf messages.
- Matching and aligning time-indexed data sequences.
- Constructing protobuf messages from processed data.
"""

import collections
import datetime
import itertools
from typing import Any, List, Mapping, Sequence, Set, Tuple, Union

from absl import logging
import gin
import numpy as np
import pandas as pd

from smart_control.models import base_occupancy # For type hinting
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import setpoint_schedule as setpoint_schedule_lib # For type hinting
from smart_control.utils import conversion_utils

# Constants for protobuf field names and prefixes, used for structuring data.
_VALUE_TYPE = smart_control_building_pb2.DeviceInfo.ValueType
_ACTION_RESPONSE_TYPE = (
    smart_control_building_pb2.SingleActionResponse.ActionResponseType
)
_ACTION_DEVICE_TYPES: List[smart_control_building_pb2.DeviceInfo.DeviceType] = [
    smart_control_building_pb2.DeviceInfo.DeviceType.AHU,
    smart_control_building_pb2.DeviceInfo.DeviceType.BLR,
    smart_control_building_pb2.DeviceInfo.DeviceType.AC, # Assuming AC is distinct from AHU
]
_ACTION_PREFIX: str = "action"
_TIMESTAMP: str = "timestamp"
_REWARD_INFO_PREFIX: str = "reward_info" # Changed from _REWARD_INFO
_START_SUFFIX: str = "start" # Changed from _START
_END_SUFFIX: str = "end" # Changed from _END
_BLOWER_ELECTRICAL_ENERGY_RATE_FIELD: str = "blower_electrical_energy_rate"
_AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE_FIELD: str = (
    "air_conditioning_electrical_energy_rate"
)
_NATURAL_GAS_HEATING_ENERGY_RATE_FIELD: str = "natural_gas_heating_energy_rate"
_PUMP_ELECTRICAL_ENERGY_RATE_FIELD: str = "pump_electrical_energy_rate"
_ZONE_AIR_TEMPERATURE_SENSOR_FIELD: str = "zone_air_temperature_sensor"
_ZONE_AIR_COOLING_TEMPERATURE_SETPOINT_FIELD: str = (
    "zone_air_cooling_temperature_setpoint"
)
_ZONE_AIR_HEATING_TEMPERATURE_SETPOINT_FIELD: str = (
    "zone_air_heating_temperature_setpoint"
)
_DAY_OF_WEEK_LABEL: str = "dow"
_HOUR_OF_DAY_LABEL: str = "hod"
_SIN_SUFFIX: str = "sin"
_COS_SUFFIX: str = "cos"


@gin.configurable
def get_consolidated_time_features(
    num_hour_of_day_features: int, num_day_of_week_features: int
) -> List[Union[str, Tuple[str, str]]]:
  """Generates a list of names for time-based features.

  This includes a basic 'timestamp' and cyclical sine/cosine features for
  hour of day and day of week. Useful for defining DataFrame columns or model
  feature sets. This function is Gin-configurable.

  Args:
    num_hour_of_day_features (int): Number of sine/cosine pairs for encoding
      the hour of the day.
    num_day_of_week_features (int): Number of sine/cosine pairs for encoding
      the day of the week.

  Returns:
    List[Union[str, Tuple[str, str]]]: A list of feature names.
    Example: `['timestamp', ('hod', 'cos_000'), ..., ('dow', 'sin_000'), ...]`
  """
  return (
      [_TIMESTAMP] +
      get_time_feature_names(n=num_hour_of_day_features, label=_HOUR_OF_DAY_LABEL) +
      get_time_feature_names(n=num_day_of_week_features, label=_DAY_OF_WEEK_LABEL)
  )


def get_time_feature_names(
    n: int, label: str = _HOUR_OF_DAY_LABEL
) -> List[Tuple[str, str]]:
  """Generates `2n` feature names for phase-shifted sine and cosine time signals.

  Creates names like `(label, 'cos_000')`, `(label, 'sin_000')`, ...,
  `(label, 'cos_NNN')`, `(label, 'sin_NNN')` where NNN is `n-1`.

  Args:
    n (int): The number of sine/cosine pairs.
    label (str): A prefix label for the feature names (e.g., "hod", "dow").
      Defaults to "hod".

  Returns:
    List[Tuple[str, str]]: A list of `2n` tuples, each representing a
    time feature name.
  """
  cos_names = [(label, f"{_COS_SUFFIX}_{i:03d}") for i in range(n)]
  sin_names = [(label, f"{_SIN_SUFFIX}_{i:03d}") for i in range(n)]
  return cos_names + sin_names


def expand_time_features(
    n: int, radian_time: float, label: str = _HOUR_OF_DAY_LABEL
) -> Mapping[Tuple[str, str], float]:
  """Expands a single radian time value into `2n` phase-shifted sine/cosine features.

  This is used to create cyclical features for time, e.g., hour of day or
  day of week, which can help models understand time's cyclical nature.
  The phase shifts are `k * 2*pi/n` for k in `0..n-1`.

  Args:
    n (int): The number of sine/cosine pairs to generate.
    radian_time (float): The base time value in radians (typically from 0 to 2*pi).
    label (str): A prefix label for the feature names (e.g., "hod", "dow").

  Returns:
    Mapping[Tuple[str, str], float]: A dictionary mapping feature name tuples
    (e.g., `(label, 'cos_000')`) to their calculated float values.
  """
  feature_names = get_time_feature_names(n, label)
  # Phase shifts: k * (2*pi / n) for k = 0, ..., n-1
  phase_shifts = (np.arange(n) / float(n)) * 2.0 * np.pi
  phases = radian_time + phase_shifts

  cos_components = np.cos(phases)
  sin_components = np.sin(phases)

  # Ensure order matches get_time_feature_names: all cosines then all sines
  expanded_values = np.concatenate([cos_components, sin_components])

  if len(feature_names) != len(expanded_values):
    raise ValueError("Mismatch between generated feature names and values.")

  return dict(zip(feature_names, expanded_values))


def get_observation_sequence(
    observation_responses: Sequence[smart_control_building_pb2.ObservationResponse],
    feature_tuples: Set[Tuple[str, str]], # (device_id, measurement_name)
    time_zone: str = "UTC",
    num_hour_features: int = 1,
    num_dow_features: int = 1,
) -> pd.DataFrame:
  """Converts a sequence of `ObservationResponse` protos to a Pandas DataFrame.

  Each row in the DataFrame corresponds to one `ObservationResponse`.
  Columns include a timestamp, cyclical time features (hour of day, day of
  week), and all features specified in `feature_tuples`.

  Args:
    observation_responses (Sequence[ObservationResponse]): A list of
      `ObservationResponse` protobuf messages.
    feature_tuples (Set[Tuple[str, str]]): A set of desired feature tuples,
      where each tuple is (device_id, measurement_name). These will form
      columns in the output DataFrame.
    time_zone (str): The IANA time zone string to which timestamps will be
      converted (e.g., "America/Los_Angeles"). Defaults to "UTC".
    num_hour_features (int): Number of sine/cosine pairs for hour-of-day features.
    num_dow_features (int): Number of sine/cosine pairs for day-of-week features.

  Returns:
    pd.DataFrame: A DataFrame containing the processed observation data.
  """
  # Define column order for the DataFrame
  columns = (
      [_TIMESTAMP] +
      get_time_feature_names(num_hour_features, _HOUR_OF_DAY_LABEL) +
      get_time_feature_names(num_dow_features, _DAY_OF_WEEK_LABEL) +
      sorted(list(feature_tuples)) # Ensure consistent column order
  )

  data_rows = []
  for obs_response in observation_responses:
    feature_map = get_feature_map(
        obs_response, time_zone, num_hour_features, num_dow_features
    )
    # Ensure all expected columns are present, fill with NaN if missing
    row_data = {col: feature_map.get(col, np.nan) for col in columns}
    data_rows.append(row_data)

  return pd.DataFrame(data_rows, columns=columns)


def get_feature_map(
    observation_response: smart_control_building_pb2.ObservationResponse,
    time_zone: str = "UTC",
    num_hour_features: int = 1,
    num_dow_features: int = 1,
) -> Dict[Any, Any]: # Column name can be str or tuple
  """Converts a single `ObservationResponse` to a feature dictionary.

  The dictionary includes the timestamp, cyclical time features, and all valid
  continuous measurements from the `ObservationResponse`.

  Args:
    observation_response (smart_control_building_pb2.ObservationResponse): The
      protobuf message to process.
    time_zone (str): The target IANA time zone for the timestamp.
    num_hour_features (int): Number of sine/cosine pairs for hour-of-day.
    num_dow_features (int): Number of sine/cosine pairs for day-of-week.

  Returns:
    Dict[Any, Any]: A dictionary where keys are feature names (str for
    timestamp, tuples like (device_id, measurement_name) for sensor readings,
    or (label, type_idx) for time features) and values are their corresponding
    values.
  """
  feature_map: Dict[Any, Any] = {}
  pd_timestamp = conversion_utils.proto_to_pandas_timestamp(
      observation_response.timestamp
  ).tz_convert(time_zone)

  feature_map[_TIMESTAMP] = pd_timestamp
  feature_map.update(expand_time_features(
      n=num_hour_features,
      radian_time=conversion_utils.get_radian_time(
          pd_timestamp, conversion_utils.TimeIntervalEnum.HOUR_OF_DAY
      ),
      label=_HOUR_OF_DAY_LABEL
  ))
  feature_map.update(expand_time_features(
      n=num_dow_features,
      radian_time=conversion_utils.get_radian_time(
          pd_timestamp, conversion_utils.TimeIntervalEnum.DAY_OF_WEEK
      ),
      label=_DAY_OF_WEEK_LABEL
  ))

  for single_obs_resp in observation_response.single_observation_responses:
    request = single_obs_resp.single_observation_request
    if single_obs_resp.observation_valid:
      feature_map[(request.device_id, request.measurement_name)] = (
          single_obs_resp.continuous_value
      )
  return feature_map


def get_action_tuples(
    action_response: smart_control_building_pb2.ActionResponse,
) -> Set[Tuple[str, str, str]]:
  """Extracts unique action identifiers from an `ActionResponse`.

  Each action identifier is a tuple `(_ACTION_PREFIX, device_id, setpoint_name)`.

  Args:
    action_response (smart_control_building_pb2.ActionResponse): The protobuf
      message containing action requests.

  Returns:
    Set[Tuple[str, str, str]]: A set of unique action identifier tuples.
  """
  action_tuples: Set[Tuple[str, str, str]] = set()
  for single_action_request in action_response.request.single_action_requests:
    action_tuples.add((
        _ACTION_PREFIX,
        single_action_request.device_id,
        single_action_request.setpoint_name
    ))
  return action_tuples


def get_feature_tuples(
    observation_response: smart_control_building_pb2.ObservationResponse,
) -> Set[Tuple[str, str]]:
  """Extracts unique (device_id, measurement_name) tuples from an `ObservationResponse`.

  Only considers valid observations.

  Args:
    observation_response (smart_control_building_pb2.ObservationResponse):
      The protobuf message.

  Returns:
    Set[Tuple[str, str]]: A set of unique (device_id, measurement_name) tuples.
  """
  feature_tuples: Set[Tuple[str, str]] = set()
  for single_obs_resp in observation_response.single_observation_responses:
    if single_obs_resp.observation_valid:
      request = single_obs_resp.single_observation_request
      feature_tuples.add((request.device_id, request.measurement_name))
  return feature_tuples


def get_action_map(
    action_response: smart_control_building_pb2.ActionResponse,
    time_zone: Union[str, datetime.tzinfo] = "UTC",
) -> Dict[Any, Any]:
  """Converts an `ActionResponse` to a dictionary mapping action tuples to values.

  Includes a timestamp. If an action was not accepted, its value is set to NaN.

  Args:
    action_response (smart_control_building_pb2.ActionResponse): The protobuf message.
    time_zone (Union[str, datetime.tzinfo]): Target time zone for the timestamp.

  Returns:
    Dict[Any, Any]: Dictionary mapping action identifiers (tuples like
    `(_ACTION_PREFIX, device_id, setpoint_name)`) or `_TIMESTAMP` (str) to
    their values.
  """
  action_map: Dict[Any, Any] = {}
  pd_timestamp = conversion_utils.proto_to_pandas_timestamp(
      action_response.timestamp
  ).tz_convert(time_zone)
  action_map[_TIMESTAMP] = pd_timestamp

  for single_action_resp in action_response.single_action_responses:
    request = single_action_resp.request
    action_key = (_ACTION_PREFIX, request.device_id, request.setpoint_name)
    if single_action_resp.response_type == _ACTION_RESPONSE_TYPE.ACCEPTED:
      action_map[action_key] = request.continuous_value
    else:
      action_map[action_key] = np.nan # Mark rejected/failed actions as NaN
  return action_map


def get_reward_info_tuples(
    reward_info: smart_control_reward_pb2.RewardInfo,
) -> Set[Tuple[str, str, str]]:
  """Extracts unique identifiers for energy-related data from `RewardInfo`.

  Identifiers are tuples like `(_REWARD_INFO_PREFIX, device_id, field_name)`
  or `(_REWARD_INFO_PREFIX, _TIMESTAMP, _START_SUFFIX)`.

  Args:
    reward_info (smart_control_reward_pb2.RewardInfo): The protobuf message.

  Returns:
    Set[Tuple[str, str, str]]: A set of unique identifier tuples for data
    points within the `RewardInfo` message.
  """
  reward_info_tuples: Set[Tuple[str, str, str]] = set()
  reward_info_tuples.add((_REWARD_INFO_PREFIX, _TIMESTAMP, _START_SUFFIX))
  reward_info_tuples.add((_REWARD_INFO_PREFIX, _TIMESTAMP, _END_SUFFIX))

  for ah_id in reward_info.air_handler_reward_infos:
    reward_info_tuples.add((_REWARD_INFO_PREFIX, ah_id, _BLOWER_ELECTRICAL_ENERGY_RATE_FIELD))
    reward_info_tuples.add((_REWARD_INFO_PREFIX, ah_id, _AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE_FIELD))
  for boiler_id in reward_info.boiler_reward_infos:
    reward_info_tuples.add((_REWARD_INFO_PREFIX, boiler_id, _NATURAL_GAS_HEATING_ENERGY_RATE_FIELD))
    reward_info_tuples.add((_REWARD_INFO_PREFIX, boiler_id, _PUMP_ELECTRICAL_ENERGY_RATE_FIELD))
  return reward_info_tuples


def get_reward_info_map(
    reward_info: smart_control_reward_pb2.RewardInfo,
    time_zone: Union[str, datetime.tzinfo] = "UTC",
) -> Dict[Tuple[str, str, str], Any]: # Value can be Timestamp or float
  """Converts a `RewardInfo` proto to a dictionary mapping identifiers to values.

  Args:
    reward_info (smart_control_reward_pb2.RewardInfo): The protobuf message.
    time_zone (Union[str, datetime.tzinfo]): Target time zone for timestamps.

  Returns:
    Dict[Tuple[str, str, str], Any]: A dictionary mapping identifier tuples
    to their corresponding values (timestamps or floats).
  """
  reward_map: Dict[Tuple[str, str, str], Any] = {}
  start_ts = conversion_utils.proto_to_pandas_timestamp(
      reward_info.start_timestamp
  ).tz_convert(time_zone)
  reward_map[(_REWARD_INFO_PREFIX, _TIMESTAMP, _START_SUFFIX)] = start_ts
  end_ts = conversion_utils.proto_to_pandas_timestamp(
      reward_info.end_timestamp
  ).tz_convert(time_zone)
  reward_map[(_REWARD_INFO_PREFIX, _TIMESTAMP, _END_SUFFIX)] = end_ts

  for ah_id, ah_data in reward_info.air_handler_reward_infos.items():
    reward_map[(_REWARD_INFO_PREFIX, ah_id, _BLOWER_ELECTRICAL_ENERGY_RATE_FIELD)] = (
        ah_data.blower_electrical_energy_rate
    )
    reward_map[(_REWARD_INFO_PREFIX, ah_id, _AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE_FIELD)] = (
        ah_data.air_conditioning_electrical_energy_rate
    )
  for boiler_id, boiler_data in reward_info.boiler_reward_infos.items():
    reward_map[(_REWARD_INFO_PREFIX, boiler_id, _NATURAL_GAS_HEATING_ENERGY_RATE_FIELD)] = (
        boiler_data.natural_gas_heating_energy_rate
    )
    reward_map[(_REWARD_INFO_PREFIX, boiler_id, _PUMP_ELECTRICAL_ENERGY_RATE_FIELD)] = (
        boiler_data.pump_electrical_energy_rate
    )
  return reward_map


def get_matching_indexes(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    step_interval_timedelta: pd.Timedelta,
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
  """Finds matching timestamp indexes between two DataFrames, offset by an interval.

  This function aligns rows from `input_df` and `output_df` such that
  if `t_in` is an index from `input_df` and `t_out` is from `output_df`,
  then `0 < t_out - t_in <= step_interval_timedelta`. It's useful for creating
  paired input-output samples for sequence models where the output is predicted
  one step ahead.

  Args:
    input_df (pd.DataFrame): DataFrame with a DatetimeIndex, representing inputs.
    output_df (pd.DataFrame): DataFrame with a DatetimeIndex, representing outputs.
    step_interval_timedelta (pd.Timedelta): The expected time delay or step
      interval between a related input and output.

  Returns:
    Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
      - A DatetimeIndex for `input_df` containing aligned input timestamps.
      - A DatetimeIndex for `output_df` containing corresponding output timestamps.
  """
  input_df_clean = input_df.dropna()
  output_df_clean = output_df.dropna()

  # Use deques for efficient popping from the front
  input_indices_deque = collections.deque(input_df_clean.index)
  output_indices_deque = collections.deque(output_df_clean.index)

  matched_input_indices: List[pd.Timestamp] = []
  matched_output_indices: List[pd.Timestamp] = []

  if not output_indices_deque: # No output indices to match
      return pd.DatetimeIndex(matched_input_indices), pd.DatetimeIndex(matched_output_indices)

  current_output_ts = output_indices_deque.popleft()
  while input_indices_deque and output_indices_deque:
    current_input_ts = input_indices_deque.popleft()
    # Advance output_ts until it's past current_input_ts
    while current_output_ts <= current_input_ts and output_indices_deque:
      current_output_ts = output_indices_deque.popleft()
    # If last output_ts was passed, it might be the one to match
    if current_output_ts <= current_input_ts and not output_indices_deque:
        break


    time_difference = current_output_ts - current_input_ts
    # Check if the pair is within the desired step interval
    if pd.Timedelta(0) < time_difference <= step_interval_timedelta:
      matched_input_indices.append(current_input_ts)
      matched_output_indices.append(current_output_ts)
      # Potentially advance current_output_ts if it can only be used once
      if output_indices_deque:
          current_output_ts = output_indices_deque.popleft()
      else:
          break # No more output timestamps

  return pd.DatetimeIndex(matched_input_indices), pd.DatetimeIndex(matched_output_indices)


def get_reward_info_sequence(
    reward_infos: Sequence[smart_control_reward_pb2.RewardInfo],
    reward_info_tuples: Set[Tuple[str, str, str]],
    time_zone: Union[str, datetime.tzinfo] = "UTC",
) -> pd.DataFrame:
  """Converts a sequence of `RewardInfo` protos to a Pandas DataFrame.

  Args:
    reward_infos (Sequence[smart_control_reward_pb2.RewardInfo]): List of
      `RewardInfo` protobuf messages.
    reward_info_tuples (Set[Tuple[str, str, str]]): A set of column names
      (tuples) that are expected in the output DataFrame, corresponding to
      data extracted from `RewardInfo`.
    time_zone (Union[str, datetime.tzinfo]): Target time zone for timestamps.

  Returns:
    pd.DataFrame: A DataFrame where each row corresponds to a `RewardInfo`
    message. Columns are sorted `reward_info_tuples`.
  """
  data_rows = [
      get_reward_info_map(ri, time_zone) for ri in reward_infos
  ]
  return pd.DataFrame(data_rows, columns=sorted(list(reward_info_tuples)))


def get_action_sequence(
    action_responses: Sequence[smart_control_building_pb2.ActionResponse],
    action_tuples: Set[Tuple[str, str, str]],
    time_zone: Union[str, datetime.tzinfo] = "UTC",
) -> pd.DataFrame:
  """Converts a sequence of `ActionResponse` protos to a Pandas DataFrame.

  Args:
    action_responses (Sequence[smart_control_building_pb2.ActionResponse]):
      List of `ActionResponse` protobuf messages.
    action_tuples (Set[Tuple[str, str, str]]): A set of column names (tuples)
      expected in the output DataFrame, corresponding to actions.
    time_zone (Union[str, datetime.tzinfo]): Target time zone for timestamps.

  Returns:
    pd.DataFrame: A DataFrame where each row corresponds to an `ActionResponse`.
    Columns include `_TIMESTAMP` and sorted `action_tuples`.
  """
  # Define column order: timestamp first, then sorted action tuples
  columns_ordered = [_TIMESTAMP] + sorted(list(action_tuples))
  data_rows: List[Dict[Any, Any]] = []

  for action_resp in action_responses:
    action_map = get_action_map(action_resp, time_zone)
    # Ensure only expected columns are included, fill missing with NaN
    row_data = {col: action_map.get(col, np.nan) for col in columns_ordered}
    data_rows.append(row_data)

  return pd.DataFrame(data_rows, columns=columns_ordered)


def get_device_action_tuples(
    devices_info: Sequence[smart_control_building_pb2.DeviceInfo],
) -> Sequence[Tuple[str, str, str]]:
  """Extracts actionable (device_id, setpoint_name) tuples from `DeviceInfo`.

  Filters for devices specified in `_ACTION_DEVICE_TYPES`.

  Args:
    devices_info (Sequence[smart_control_building_pb2.DeviceInfo]): A list of
      `DeviceInfo` protobuf messages.

  Returns:
    Sequence[Tuple[str, str, str]]: A list of tuples, where each tuple is
    `(_ACTION_PREFIX, device_id, setpoint_name)`.
  """
  device_action_tuples_list: List[Tuple[str, str, str]] = []
  for dev_info in devices_info:
    if dev_info.device_type in _ACTION_DEVICE_TYPES:
      for action_name in dev_info.action_fields: # action_fields is a map
        device_action_tuples_list.append(
            (_ACTION_PREFIX, dev_info.device_id, action_name)
        )
  return device_action_tuples_list


def get_observation_response(
    observation_request: smart_control_building_pb2.ObservationRequest,
    native_observations_map: Mapping[Tuple[str, str], float], # (dev_id, meas_name) -> val
    current_pd_timestamp: pd.Timestamp,
) -> smart_control_building_pb2.ObservationResponse:
  """Constructs an `ObservationResponse` proto from a request and current values.

  Args:
    observation_request (smart_control_building_pb2.ObservationRequest):
      The original request defining which observations are needed.
    native_observations_map (Mapping[Tuple[str, str], float]): A dictionary
      mapping (device_id, measurement_name) tuples to their current native
      float values.
    current_pd_timestamp (pd.Timestamp): The timestamp for this observation.

  Returns:
    smart_control_building_pb2.ObservationResponse: A populated
    `ObservationResponse` message.
  """
  obs_response = smart_control_building_pb2.ObservationResponse()
  obs_response.request.CopyFrom(observation_request)
  obs_response.timestamp.CopyFrom(
      conversion_utils.pandas_to_proto_timestamp(current_pd_timestamp)
  )

  for single_req in observation_request.single_observation_requests:
    single_resp = obs_response.single_observation_responses.add()
    single_resp.single_observation_request.CopyFrom(single_req)
    single_resp.timestamp.CopyFrom(obs_response.timestamp) # Use overall response ts

    observation_key = (single_req.device_id, single_req.measurement_name)
    if observation_key in native_observations_map:
      single_resp.continuous_value = native_observations_map[observation_key]
      single_resp.observation_valid = True
    else:
      single_resp.observation_valid = False # Value not available
      # Consider logging a warning if a requested observation is missing.
  return obs_response


def observation_response_to_observation_mapping(
    observation_response: smart_control_building_pb2.ObservationResponse,
) -> Mapping[Tuple[str, str], float]:
  """Converts an `ObservationResponse` to a dictionary of valid observations.

  Args:
    observation_response (smart_control_building_pb2.ObservationResponse):
      The protobuf message to process.

  Returns:
    Mapping[Tuple[str, str], float]: A dictionary where keys are
    (device_id, measurement_name) tuples and values are their corresponding
    continuous float values, for valid observations only.
  """
  native_obs_map: Dict[Tuple[str, str], float] = {}
  for single_obs_resp in observation_response.single_observation_responses:
    if single_obs_resp.observation_valid:
      request = single_obs_resp.single_observation_request
      native_obs_map[(request.device_id, request.measurement_name)] = (
          single_obs_resp.continuous_value
      )
  return native_obs_map


def create_action_response(
    action_request: smart_control_building_pb2.ActionRequest,
    current_pd_timestamp: pd.Timestamp,
    valid_device_action_tuples: Sequence[Tuple[str, str, str]],
) -> smart_control_building_pb2.ActionResponse:
  """Creates an `ActionResponse` based on an `ActionRequest` and valid actions.

  Marks actions as ACCEPTED if they are in `valid_device_action_tuples`,
  otherwise as REJECTED_INVALID_DEVICE.

  Args:
    action_request (smart_control_building_pb2.ActionRequest): The incoming
      request from the agent.
    current_pd_timestamp (pd.Timestamp): The current simulation timestamp to
      record in the response.
    valid_device_action_tuples (Sequence[Tuple[str, str, str]]): A sequence of
      valid action tuples `(_ACTION_PREFIX, device_id, setpoint_name)` that
      the environment can process.

  Returns:
    smart_control_building_pb2.ActionResponse: A populated `ActionResponse`.
  """
  action_response_proto = smart_control_building_pb2.ActionResponse()
  action_response_proto.request.CopyFrom(action_request)
  action_response_proto.timestamp.CopyFrom(
      conversion_utils.pandas_to_proto_timestamp(current_pd_timestamp)
  )

  valid_actions_set = set(valid_device_action_tuples) # For efficient lookup

  for single_req in action_request.single_action_requests:
    single_resp = action_response_proto.single_action_responses.add()
    single_resp.request.CopyFrom(single_req)
    action_tuple_key = (
        _ACTION_PREFIX, single_req.device_id, single_req.setpoint_name
    )
    if action_tuple_key in valid_actions_set:
      single_resp.response_type = _ACTION_RESPONSE_TYPE.ACCEPTED
    else:
      single_resp.response_type = _ACTION_RESPONSE_TYPE.REJECTED_INVALID_DEVICE
      logging.warning("Rejected invalid action: %s", action_tuple_key)
  return action_response_proto


def split_output_into_observations_and_reward_info_mapping(
    model_output_map: Mapping[Tuple[str, ...], float],
) -> Tuple[Mapping[Tuple[str, str], float], Mapping[Tuple[str, str, str], float]]:
  """Separates a combined model output map into observation and reward components.

  Assumes model output keys are tuples, where the first element indicates if
  it's a reward-related info (`_REWARD_INFO_PREFIX`) or an observation.

  Args:
    model_output_map (Mapping[Tuple[str, ...], float]): A flat dictionary where
      keys are tuples identifying the data point (e.g.,
      (_REWARD_INFO_PREFIX, device, field) or (device, field)) and values are floats.

  Returns:
    Tuple[Mapping[Tuple[str, str], float], Mapping[Tuple[str, str, str], float]]:
      - A mapping for observations: `{(device_id, measurement_name): value}`.
      - A mapping for reward info: `{(_REWARD_INFO_PREFIX, device_id, field): value}`.
  """
  reward_info_sub_map: Dict[Tuple[str, str, str], float] = {}
  observation_sub_map: Dict[Tuple[str, str], float] = {}

  for key_tuple, value in model_output_map.items():
    if key_tuple and key_tuple[0] == _REWARD_INFO_PREFIX:
      # Ensure key has 3 parts for reward info (prefix, device, field)
      if len(key_tuple) == 3:
        reward_info_sub_map[key_tuple] = value # type: ignore
      else:
        logging.warning("Malformed reward info key in output map: %s", key_tuple)
    else:
      # Ensure key has 2 parts for observation (device, field)
      if len(key_tuple) == 2:
        observation_sub_map[key_tuple] = value # type: ignore
      else:
        logging.warning("Malformed observation key in output map: %s", key_tuple)
  return observation_sub_map, reward_info_sub_map


def get_reward_info_devices(
    reward_info_flat_map: Mapping[Tuple[str, str, str], float],
) -> Mapping[str, Mapping[str, float]]:
  """Restructures a flat reward info map into a nested map by device ID.

  Args:
    reward_info_flat_map (Mapping[Tuple[str, str, str], float]): A flat map
      where keys are `(_REWARD_INFO_PREFIX, device_id, field_name)` and
      values are floats.

  Returns:
    Mapping[str, Mapping[str, float]]: A nested map:
    `{device_id: {field_name: value}}`.
  """
  device_centric_map = collections.defaultdict(dict)
  for (_prefix, device_id, field_name), value in reward_info_flat_map.items():
    if _prefix == _REWARD_INFO_PREFIX: # Ensure it's actually reward info
      device_centric_map[device_id][field_name] = value
  return device_centric_map


def action_request_to_action_mapping(
    action_request: smart_control_building_pb2.ActionRequest,
    valid_device_action_tuples: Sequence[Tuple[str, str, str]],
) -> Mapping[Tuple[str, str, str], float]:
  """Converts an `ActionRequest` proto to a dictionary of valid actions.

  Only actions present in `valid_device_action_tuples` are included.
  Logs a warning if the request contains actions not in the valid set or
  if it doesn't set all actions defined in `valid_device_action_tuples`.

  Args:
    action_request (smart_control_building_pb2.ActionRequest): The protobuf
      message from the agent.
    valid_device_action_tuples (Sequence[Tuple[str, str, str]]): A sequence of
      tuples `(_ACTION_PREFIX, device_id, setpoint_name)` representing all
      actions the environment expects/can handle.

  Returns:
    Mapping[Tuple[str, str, str], float]: A dictionary where keys are valid
    action tuples and values are their corresponding float values from the request.
  """
  device_action_map: Dict[Tuple[str, str, str], float] = {}
  requested_actions_set: Set[Tuple[str, str, str]] = set()

  for single_req in action_request.single_action_requests:
    action_tuple = (
        _ACTION_PREFIX, single_req.device_id, single_req.setpoint_name
    )
    requested_actions_set.add(action_tuple)
    if action_tuple in valid_device_action_tuples:
      device_action_map[action_tuple] = single_req.continuous_value
    else:
      logging.warning("ActionRequest contained an invalid/unknown action tuple: %s", action_tuple)

  # Check if all valid actions were set by the agent
  missing_actions = set(valid_device_action_tuples) - requested_actions_set
  if missing_actions:
    logging.warning(
        "Agent's ActionRequest did not set %d expected actions: %s",
        len(missing_actions), missing_actions
    )
  return device_action_map


def get_boiler_reward_infos(
    reward_info_by_device: Mapping[str, Mapping[str, float]],
) -> Mapping[str, smart_control_reward_pb2.RewardInfo.BoilerRewardInfo]:
  """Creates `BoilerRewardInfo` protos from a device-centric reward data map.

  Identifies boilers by checking for the presence of specific energy rate fields.

  Args:
    reward_info_by_device (Mapping[str, Mapping[str, float]]): A nested map:
      `{device_id: {field_name: value}}`.

  Returns:
    Mapping[str, smart_control_reward_pb2.RewardInfo.BoilerRewardInfo]: A map
    from boiler device IDs to their `BoilerRewardInfo` protobuf messages.
  """
  boiler_reward_infos_map: Dict[str, smart_control_reward_pb2.RewardInfo.BoilerRewardInfo] = {}
  for device_id, fields in reward_info_by_device.items():
    gas_rate = fields.get(_NATURAL_GAS_HEATING_ENERGY_RATE_FIELD)
    pump_rate = fields.get(_PUMP_ELECTRICAL_ENERGY_RATE_FIELD)

    # A device is considered a boiler if it reports both these fields.
    if gas_rate is not None and pump_rate is not None:
      boiler_reward_infos_map[device_id] = (
          smart_control_reward_pb2.RewardInfo.BoilerRewardInfo(
              natural_gas_heating_energy_rate=gas_rate,
              pump_electrical_energy_rate=pump_rate,
          )
      )
  return boiler_reward_infos_map


def get_air_handler_reward_infos(
    reward_info_by_device: Mapping[str, Mapping[str, float]],
) -> Mapping[str, smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo]:
  """Creates `AirHandlerRewardInfo` protos from a device-centric reward data map.

  Identifies air handlers by checking for specific energy rate fields.

  Args:
    reward_info_by_device (Mapping[str, Mapping[str, float]]): A nested map:
      `{device_id: {field_name: value}}`.

  Returns:
    Mapping[str, smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo]: A map
    from air handler device IDs to their `AirHandlerRewardInfo` protos.
  """
  ah_reward_infos_map: Dict[str, smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo] = {}
  for device_id, fields in reward_info_by_device.items():
    blower_rate = fields.get(_BLOWER_ELECTRICAL_ENERGY_RATE_FIELD)
    ac_rate = fields.get(_AIR_CONDITIONING_ELECTRICAL_ENERGY_RATE_FIELD)

    # An AHU must report blower and AC energy rates.
    if blower_rate is not None and ac_rate is not None:
      ah_reward_infos_map[device_id] = (
          smart_control_reward_pb2.RewardInfo.AirHandlerRewardInfo(
              blower_electrical_energy_rate=blower_rate,
              air_conditioning_electrical_energy_rate=ac_rate,
          )
      )
  return ah_reward_infos_map


def get_current_device_observations(
    current_observations_map: Mapping[Tuple[str, str], float],
    target_device_id: str
) -> Mapping[str, float]:
  """Filters observations for a specific device.

  Args:
    current_observations_map (Mapping[Tuple[str, str], float]): A flat map of
      all current observations: `{(device_id, measurement_name): value}`.
    target_device_id (str): The ID of the device for which to retrieve observations.

  Returns:
    Mapping[str, float]: A dictionary `{measurement_name: value}` for the
    specified `target_device_id`.
  """
  return {
      measurement_name: value
      for (dev_id, measurement_name), value in current_observations_map.items()
      if dev_id == target_device_id
  }


def get_zone_reward_infos(
    current_pd_timestamp: pd.Timestamp,
    step_interval_timedelta: pd.Timedelta,
    current_observations_map: Mapping[Tuple[str, str], float],
    occupancy_model: base_occupancy.BaseOccupancy,
    thermostat_setpoint_schedule: setpoint_schedule_lib.SetpointSchedule,
    zone_info_list: Sequence[smart_control_building_pb2.ZoneInfo],
    device_info_list: Sequence[smart_control_building_pb2.DeviceInfo],
) -> Mapping[str, smart_control_reward_pb2.RewardInfo.ZoneRewardInfo]:
  """Constructs `ZoneRewardInfo` messages for all zones.

  This function gathers all necessary data (temperatures, setpoints, occupancy)
  for each zone to populate its `ZoneRewardInfo` proto.

  Args:
    current_pd_timestamp (pd.Timestamp): The current simulation timestamp.
    step_interval_timedelta (pd.Timedelta): Duration of the simulation step.
    current_observations_map (Mapping[Tuple[str, str], float]): Flat map of
      current sensor readings `{(device_id, measurement_name): value}`.
    occupancy_model (base_occupancy.BaseOccupancy): Model to get zone occupancy.
    thermostat_setpoint_schedule (setpoint_schedule_lib.SetpointSchedule): Schedule
      for thermostat setpoints.
    zone_info_list (Sequence[smart_control_building_pb2.ZoneInfo]): List of
      `ZoneInfo` protos for all zones.
    device_info_list (Sequence[smart_control_building_pb2.DeviceInfo]): List of
      `DeviceInfo` protos for all devices.

  Returns:
    Mapping[str, smart_control_reward_pb2.RewardInfo.ZoneRewardInfo]: A map
    from zone IDs (str) to their populated `ZoneRewardInfo` messages.

  Raises:
    ValueError: If heating setpoint is found to be greater than cooling setpoint
      from the schedule.
  """
  zone_reward_infos_map: Dict[str, smart_control_reward_pb2.RewardInfo.ZoneRewardInfo] = {}
  # Create a quick lookup for devices in each zone
  zone_to_device_ids_map = {
      zi.zone_id: list(zi.devices) for zi in zone_info_list
  }
  # Create a quick lookup for device info by ID
  device_id_to_info_map = {di.device_id: di for di in device_info_list}

  # Get current setpoints from schedule (applies to all zones generally)
  # Individual VAVs might have their own thermostats that could override this,
  # but for reward calculation, we often use the scheduled setpoints as reference.
  (
      scheduled_heating_sp_k, scheduled_cooling_sp_k
  ) = thermostat_setpoint_schedule.get_temperature_window(current_pd_timestamp)

  if scheduled_heating_sp_k > scheduled_cooling_sp_k:
    raise ValueError(
        f"Heating setpoint {scheduled_heating_sp_k}K cannot be greater than "
        f"cooling setpoint {scheduled_cooling_sp_k}K."
    )

  for zone_info_proto in zone_info_list:
    zone_id_str = zone_info_proto.zone_id
    avg_occupancy = occupancy_model.average_zone_occupancy(
        zone_id=zone_id_str,
        start_time=current_pd_timestamp - step_interval_timedelta,
        end_time=current_pd_timestamp,
    )

    # Find the VAV and its temperature sensor for this zone
    zone_temp_k: Optional[float] = None
    # These might remain the scheduled ones if no specific VAV setpoints are found
    actual_heating_sp_k = scheduled_heating_sp_k
    actual_cooling_sp_k = scheduled_cooling_sp_k

    for dev_id_in_zone in zone_to_device_ids_map.get(zone_id_str, []):
      device_info = device_id_to_info_map.get(dev_id_in_zone)
      if device_info and device_info.device_type == _VALUE_TYPE.VAV:
        device_obs = get_current_device_observations(
            current_observations_map, dev_id_in_zone
        )
        if _ZONE_AIR_TEMPERATURE_SENSOR_FIELD in device_obs:
          zone_temp_k = device_obs[_ZONE_AIR_TEMPERATURE_SENSOR_FIELD]
        # Optionally, try to get actual VAV setpoints if they are observable
        if _ZONE_AIR_HEATING_TEMPERATURE_SETPOINT_FIELD in device_obs:
          actual_heating_sp_k = device_obs[_ZONE_AIR_HEATING_TEMPERATURE_SETPOINT_FIELD]
        if _ZONE_AIR_COOLING_TEMPERATURE_SETPOINT_FIELD in device_obs:
          actual_cooling_sp_k = device_obs[_ZONE_AIR_COOLING_TEMPERATURE_SETPOINT_FIELD]
        break # Assume one VAV per zone for temperature sensing for now

    if zone_temp_k is not None:
      zone_reward_infos_map[zone_id_str] = (
          smart_control_reward_pb2.RewardInfo.ZoneRewardInfo(
              heating_setpoint_temperature=actual_heating_sp_k,
              cooling_setpoint_temperature=actual_cooling_sp_k,
              zone_air_temperature=zone_temp_k,
              average_occupancy=avg_occupancy,
          )
      )
    else:
      logging.warning("No zone air temperature sensor found for zone: %s", zone_id_str)

  return zone_reward_infos_map
