"""Utility functions for processing and transforming reinforcement learning data.

This module provides a collection of helper functions designed to extract,
convert, and structure data collected during reinforcement learning episodes.
These functions typically process lists of protobuf messages (like RewardInfo,
ObservationResponse, ActionResponse from the smart building environment) and
transform them into pandas DataFrames or Series, which are more convenient for
analysis, plotting, and further processing.
"""

import logging
import os
from typing import Any, List, Union # Removed Sequence, using List consistently

import numpy as np
import pandas as pd

# Assuming RewardInfo, ObservationResponse, ActionResponse protos are defined elsewhere
# For type hinting, it would be better to import them directly, e.g.:
# from smart_control.proto import smart_control_reward_pb2
# from smart_control.proto import smart_control_building_pb2
# However, using 'Any' for now if direct imports are too complex for this context.

from smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE
from smart_control.reinforcement_learning.utils.constants import KELVIN_TO_CELSIUS
from smart_control.utils import controller_reader # For ProtoReader type hint
from smart_control.utils import conversion_utils

logger = logging.getLogger(__name__)


def get_latest_episode_reader(
    metrics_path: str,
) -> controller_reader.ProtoReader:
  """Initializes a ProtoReader for the most recent episode in a metrics directory.

  This function scans the specified `metrics_path` for subdirectories, which
  are assumed to represent individual episodes and are named in a way that
  allows chronological sorting (e.g., timestamp-based names). It then selects
  the latest episode directory and returns a `ProtoReader` instance pointing
  to it.

  Args:
    metrics_path: The file system path to the parent directory containing
      subdirectories for one or more episodes.

  Returns:
    A `controller_reader.ProtoReader` instance initialized for reading data
    from the most recent episode found within `metrics_path`.

  Raises:
    IndexError: If no episode directories are found in `metrics_path`.
    FileNotFoundError: If `metrics_path` does not exist.
  """
  logger.info("Scanning for latest episode in: %s", metrics_path)
  episode_infos = controller_reader.get_episode_data(metrics_path).sort_index()
  if episode_infos.empty:
    raise IndexError(f"No episode data found in metrics path: {metrics_path}")
  selected_episode = episode_infos.index[-1]
  episode_path = os.path.join(metrics_path, selected_episode)
  logger.info("Selected latest episode: %s", selected_episode)
  reader = controller_reader.ProtoReader(episode_path)
  return reader


def get_energy_timeseries(
    reward_infos: List[Any], # Should be List[smart_control_reward_pb2.RewardInfo]
    time_zone: str = DEFAULT_TIME_ZONE
) -> pd.DataFrame:
  """Extracts energy consumption rates into a time-series DataFrame.

  Processes a list of `RewardInfo` protobuf messages to create a DataFrame
  detailing various energy consumption rates (e.g., AHU blower, AHU air
  conditioner, boiler natural gas, boiler pump) for each device over time.

  Each `RewardInfo` message corresponds to a time interval. The function assumes
  a fixed interval duration (300 seconds) preceding the `end_timestamp` in
  each `RewardInfo`.

  Args:
    reward_infos: A list of `RewardInfo` protobuf objects.
    time_zone: The target timezone (e.g., 'US/Pacific') to which timestamps
      will be localized.

  Returns:
    A pandas DataFrame where each row represents an energy consumption record
    for a specific device at a given time interval. Columns include:
    - 'start_time': Localized start timestamp of the interval.
    - 'end_time': Localized end timestamp of the interval.
    - 'device_id': Identifier of the device.
    - 'device_type': Type of the device (e.g., 'air_handler', 'boiler').
    - 'air_handler_blower_electrical_energy_rate': Power rate for AHU blower.
    - 'air_handler_air_conditioner_energy_rate': Power rate for AHU A/C.
    - 'boiler_natural_gas_heating_energy_rate': Power rate for boiler gas.
    - 'boiler_pump_electrical_energy_rate': Power rate for boiler pump.
    The DataFrame is sorted by 'start_time'.
  """
  data_records = []

  for reward_info in reward_infos:
    end_timestamp = conversion_utils.proto_to_pandas_timestamp(
        reward_info.end_timestamp # pytype: disable=attribute-error
    ).tz_convert(time_zone)
    # Assuming a fixed 300-second (5 minute) interval for each RewardInfo
    start_timestamp = end_timestamp - pd.Timedelta(seconds=300)

    # Process air handler data
    for air_handler_id, ah_info in reward_info.air_handler_reward_infos.items(): # pytype: disable=attribute-error
      data_records.append({
          'start_time': start_timestamp,
          'end_time': end_timestamp,
          'device_id': air_handler_id,
          'device_type': 'air_handler',
          'air_handler_blower_electrical_energy_rate': ah_info.blower_electrical_energy_rate,
          'air_handler_air_conditioner_energy_rate': ah_info.air_conditioning_electrical_energy_rate,
          'boiler_natural_gas_heating_energy_rate': 0.0,
          'boiler_pump_electrical_energy_rate': 0.0,
      })

    # Process boiler data
    for boiler_id, b_info in reward_info.boiler_reward_infos.items(): # pytype: disable=attribute-error
      data_records.append({
          'start_time': start_timestamp,
          'end_time': end_timestamp,
          'device_id': boiler_id,
          'device_type': 'boiler',
          'air_handler_blower_electrical_energy_rate': 0.0,
          'air_handler_air_conditioner_energy_rate': 0.0,
          'boiler_natural_gas_heating_energy_rate': b_info.natural_gas_heating_energy_rate,
          'boiler_pump_electrical_energy_rate': b_info.pump_electrical_energy_rate,
      })

  if not data_records:
    # Return an empty DataFrame with specified columns if no data
    return pd.DataFrame(columns=[
        'start_time', 'end_time', 'device_id', 'device_type',
        'air_handler_blower_electrical_energy_rate',
        'air_handler_air_conditioner_energy_rate',
        'boiler_natural_gas_heating_energy_rate',
        'boiler_pump_electrical_energy_rate'
    ])

  return pd.DataFrame(data_records).sort_values(by='start_time').reset_index(drop=True)


def get_outside_air_temperature_timeseries(
    observation_responses: List[Any], # Should be List[smart_control_building_pb2.ObservationResponse]
    time_zone: str = DEFAULT_TIME_ZONE,
) -> pd.Series:
  """Extracts outside air temperature readings into a time-series.

  Parses a list of `ObservationResponse` protobufs, filters for the
  'outside_air_temperature_sensor' measurement, and returns a pandas Series
  with a DatetimeIndex (localized to `time_zone`) and the corresponding
  temperature values.

  Args:
    observation_responses: A list of `ObservationResponse` protobuf objects.
    time_zone: The target timezone for the resulting Series's DatetimeIndex.

  Returns:
    A pandas Series containing outside air temperature readings, indexed by
    localized timestamps. Returns an empty Series if no relevant data is found.
  """
  temps_data = []
  for obs_response in observation_responses:
    # obs_response is expected to have 'single_observation_responses' and 'timestamp' (for each sor)
    for sor in obs_response.single_observation_responses: # pytype: disable=attribute-error
      if sor.single_observation_request.measurement_name == 'outside_air_temperature_sensor':
        timestamp = conversion_utils.proto_to_pandas_timestamp(
            sor.timestamp # pytype: disable=attribute-error
        ).tz_convert(time_zone)
        temps_data.append({'timestamp': timestamp, 'temperature': sor.continuous_value})
        break # Assuming one OAT sensor reading per ObservationResponse message

  if not temps_data:
    return pd.Series(dtype=float)

  df = pd.DataFrame(temps_data)
  return pd.Series(df['temperature'].values, index=df['timestamp']).sort_index()


def get_reward_timeseries(
    reward_infos: List[Any], # Should be List[smart_control_reward_pb2.RewardInfo]
    reward_responses: List[Any], # Should be List[smart_control_reward_pb2.RewardResponse]
    time_zone: str = DEFAULT_TIME_ZONE,
) -> pd.DataFrame:
  """Constructs a DataFrame of reward-related metrics over time.

  This function processes lists of `RewardInfo` and `RewardResponse`
  protobufs to create a consolidated time-series DataFrame. The DataFrame
  includes the agent's reward, electricity cost, carbon emissions, total
  building occupancy, and the cumulative agent reward.

  Args:
    reward_infos: A list of `RewardInfo` protobuf objects, providing data like
      timestamps and occupancy.
    reward_responses: A list of `RewardResponse` protobuf objects, providing
      calculated rewards, costs, and emissions.
    time_zone: The target timezone for localizing timestamps in the DataFrame's
      index.

  Returns:
    A pandas DataFrame indexed by localized start timestamps of each reward
    interval. Columns include:
    - 'agent_reward_value': The reward received by the agent.
    - 'electricity_energy_cost': Cost of electricity.
    - 'carbon_emitted': Amount of carbon emitted.
    - 'occupancy': Total number of occupants in the building.
    - 'cumulative_reward': Cumulative sum of 'agent_reward_value'.
    The DataFrame is sorted by timestamp.
  """
  data_records = []
  for i in range(min(len(reward_responses), len(reward_infos))):
    reward_info = reward_infos[i]
    reward_response = reward_responses[i]

    start_timestamp = conversion_utils.proto_to_pandas_timestamp(
        reward_info.start_timestamp # pytype: disable=attribute-error
    ).tz_convert(time_zone)

    # Calculate total occupancy for this RewardInfo
    total_occupancy = sum(
        zone_info.average_occupancy
        for zone_info in reward_info.zone_reward_infos.values() # pytype: disable=attribute-error
    )

    data_records.append({
        'timestamp': start_timestamp,
        'agent_reward_value': reward_response.agent_reward_value, # pytype: disable=attribute-error
        'electricity_energy_cost': reward_response.electricity_energy_cost, # pytype: disable=attribute-error
        'carbon_emitted': reward_response.carbon_emitted, # pytype: disable=attribute-error
        'occupancy': total_occupancy,
    })

  if not data_records:
    return pd.DataFrame(columns=[
        'agent_reward_value', 'electricity_energy_cost', 'carbon_emitted',
        'occupancy', 'cumulative_reward'
    ])

  df = pd.DataFrame(data_records).set_index('timestamp').sort_index()
  df['cumulative_reward'] = df['agent_reward_value'].cumsum()
  return df


def get_zone_timeseries(
    reward_infos: List[Any], # Should be List[smart_control_reward_pb2.RewardInfo]
    time_zone: str = DEFAULT_TIME_ZONE
) -> pd.DataFrame:
  """Extracts detailed zone-level data into a time-series DataFrame.

  Processes a list of `RewardInfo` protobufs to create a DataFrame where
  each row represents the state of a specific zone during a time interval.

  Args:
    reward_infos: A list of `RewardInfo` protobuf objects.
    time_zone: The target timezone for localizing timestamps.

  Returns:
    A pandas DataFrame with columns:
    - 'start_time': Localized start timestamp of the interval.
    - 'end_time': Localized end timestamp of the interval.
    - 'zone': The identifier of the zone.
    - 'heating_setpoint_temperature': Heating setpoint for the zone.
    - 'cooling_setpoint_temperature': Cooling setpoint for the zone.
    - 'zone_air_temperature': Measured air temperature in the zone.
    - 'air_flow_rate_setpoint': Airflow rate setpoint for the zone.
    - 'air_flow_rate': Actual airflow rate in the zone.
    - 'average_occupancy': Average occupancy in the zone for the interval.
    The DataFrame is sorted by 'start_time'.
  """
  data_records = []
  for reward_info in reward_infos:
    # Assuming a fixed 300-second (5 minute) interval for each RewardInfo
    end_timestamp = conversion_utils.proto_to_pandas_timestamp(
        reward_info.end_timestamp # pytype: disable=attribute-error
    ).tz_convert(time_zone)
    start_timestamp = end_timestamp - pd.Timedelta(seconds=300)

    for zone_id, zone_info in reward_info.zone_reward_infos.items(): # pytype: disable=attribute-error
      data_records.append({
          'start_time': start_timestamp,
          'end_time': end_timestamp,
          'zone': zone_id,
          'heating_setpoint_temperature': zone_info.heating_setpoint_temperature,
          'cooling_setpoint_temperature': zone_info.cooling_setpoint_temperature,
          'zone_air_temperature': zone_info.zone_air_temperature,
          'air_flow_rate_setpoint': zone_info.air_flow_rate_setpoint,
          'air_flow_rate': zone_info.air_flow_rate,
          'average_occupancy': zone_info.average_occupancy,
      })

  if not data_records:
    return pd.DataFrame(columns=[ # Return empty DataFrame with correct columns
        'start_time', 'end_time', 'zone', 'heating_setpoint_temperature',
        'cooling_setpoint_temperature', 'zone_air_temperature',
        'air_flow_rate_setpoint', 'air_flow_rate', 'average_occupancy'
    ])

  return pd.DataFrame(data_records).sort_values(by=['start_time', 'zone']).reset_index(drop=True)


def get_action_timeseries(
    action_responses: List[Any] # Should be List[smart_control_building_pb2.ActionResponse]
) -> pd.DataFrame:
  """Extracts agent action data into a time-series DataFrame.

  Parses a list of `ActionResponse` protobufs to create a DataFrame detailing
  each action requested by the agent, including the timestamp, device, setpoint,
  requested value, and the response type from the environment.

  Args:
    action_responses: A list of `ActionResponse` protobuf objects.

  Returns:
    A pandas DataFrame where each row represents a single action request.
    Columns include:
    - 'timestamp': Timestamp of the action request (localized to UTC by default
      from `proto_to_pandas_timestamp` if not otherwise specified by proto).
    - 'device_id': Identifier of the targeted device.
    - 'setpoint_name': Name of the setpoint being commanded.
    - 'setpoint_value': The value requested for the setpoint.
    - 'response_type': The status of the action (e.g., ACCEPTED, REJECTED).
  """
  data_records = []
  for action_response in action_responses:
    timestamp = conversion_utils.proto_to_pandas_timestamp(
        action_response.timestamp # pytype: disable=attribute-error
    ) # Timestamps from proto are typically UTC
    for single_action_response in action_response.single_action_responses: # pytype: disable=attribute-error
      request = single_action_response.request # pytype: disable=attribute-error
      data_records.append({
          'timestamp': timestamp,
          'device_id': request.device_id,
          'setpoint_name': request.setpoint_name,
          'setpoint_value': request.continuous_value, # Assuming continuous value for simplicity
          'response_type': single_action_response.response_type,
      })

  if not data_records:
    return pd.DataFrame(columns=[ # Return empty DataFrame with correct columns
        'timestamp', 'device_id', 'setpoint_name', 'setpoint_value', 'response_type'
    ])
  return pd.DataFrame(data_records).sort_values(by='timestamp').reset_index(drop=True)


def convert_kelvin_to_celsius(
    temperature_kelvin: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
  """Converts temperature from Kelvin to Celsius.

  Args:
    temperature_kelvin: Temperature value(s) in Kelvin. Can be a scalar,
      NumPy array, or pandas Series.

  Returns:
    The corresponding temperature value(s) in Celsius. The return type
    matches the input type.

  Example:
    >>> convert_kelvin_to_celsius(273.15)
    0.0
    >>> convert_kelvin_to_celsius(np.array([283.15, 293.15]))
    array([10., 20.])
  """
  return temperature_kelvin - KELVIN_TO_CELSIUS


def convert_celsius_to_kelvin(
    temperature_celsius: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
  """Converts temperature from Celsius to Kelvin.

  Args:
    temperature_celsius: Temperature value(s) in Celsius. Can be a scalar,
      NumPy array, or pandas Series.

  Returns:
    The corresponding temperature value(s) in Kelvin. The return type
    matches the input type.

  Example:
    >>> convert_celsius_to_kelvin(0.0)
    273.15
    >>> convert_celsius_to_kelvin(np.array([10., 20.]))
    array([283.15, 293.15])
  """
  return temperature_celsius + KELVIN_TO_CELSIUS
