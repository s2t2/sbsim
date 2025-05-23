"""Utility functions for configuring RL environments using Gin.

This module provides helper functions that are primarily intended to be used
within Gin configuration files (.gin). These functions simplify the process of:
- Converting string representations of dates and times into Pandas objects.
- Creating instances of `ContinuousVariableInfo` for observation normalization.
- Creating instances of `BoundedActionNormalizer` for action normalization.

By making these functions Gin-configurable, users can easily define complex
environment parameters and normalizer settings within their Gin files without
needing to write extensive Python code for setup.
"""

import gin
import pandas as pd

from smart_control.proto import smart_control_normalization_pb2
from smart_control.utils import bounded_action_normalizer


@gin.configurable
def to_timestamp(date_str: str) -> pd.Timestamp:
  """Converts a date string to a Pandas Timestamp object.

  This function is a Gin-configurable utility, allowing date strings in Gin
  files to be automatically converted to `pd.Timestamp` objects when the
  configuration is parsed.

  Args:
    date_str (str): A string representing a date and/or time that can be
      parsed by `pd.Timestamp`.

  Returns:
    pd.Timestamp: The parsed Pandas Timestamp object.

  Example (in a .gin file):
    ```gin
    my_component.start_date = @to_timestamp("2023-01-01 00:00:00 PST")
    ```
  """
  return pd.Timestamp(date_str)


@gin.configurable
def local_time(time_str: str) -> pd.Timedelta:
  """Converts a time string to a Pandas Timedelta object.

  This Gin-configurable utility allows time duration strings in Gin files
  (e.g., "6 hours", "30 minutes") to be converted to `pd.Timedelta` objects.

  Args:
    time_str (str): A string representing a time duration that can be
      parsed by `pd.Timedelta` (e.g., "06:00:00", "2 days", "1H30M").

  Returns:
    pd.Timedelta: The parsed Pandas Timedelta object.

  Example (in a .gin file):
    ```gin
    my_component.event_duration = @local_time("2 hours 30 minutes")
    ```
  """
  return pd.Timedelta(time_str)


@gin.configurable
def set_observation_normalization_constants(
    field_id: str, sample_mean: float, sample_variance: float
) -> smart_control_normalization_pb2.ContinuousVariableInfo:
  """Creates a `ContinuousVariableInfo` protobuf message for observation normalization.

  This Gin-configurable function is a convenient way to define normalization
  parameters (mean and variance) for a specific observation field within a
  Gin configuration file.

  Args:
    field_id (str): The unique identifier or name of the observation field
      (e.g., "zone_air_temperature_sensor").
    sample_mean (float): The sample mean of the observation field, used for
      standard score normalization (z-score).
    sample_variance (float): The sample variance of the observation field,
      used for standard score normalization.

  Returns:
    smart_control_normalization_pb2.ContinuousVariableInfo: A protobuf message
    populated with the provided normalization constants.

  Example (in a .gin file):
    ```gin
    StandardScoreObservationNormalizer.normalization_parameters = {
      'zone_temp': @set_observation_normalization_constants(
          field_id='zone_air_temperature_sensor',
          sample_mean=295.0,  # Kelvin
          sample_variance=4.0
      )
    }
    ```
  """
  return smart_control_normalization_pb2.ContinuousVariableInfo(
      id=field_id, sample_mean=sample_mean, sample_variance=sample_variance
  )


@gin.configurable
def set_action_normalization_constants(
    min_native_value: float,
    max_native_value: float,
    min_normalized_value: float = -1.0,
    max_normalized_value: float = 1.0,
) -> bounded_action_normalizer.BoundedActionNormalizer:
  """Creates a `BoundedActionNormalizer` instance for action normalization.

  This Gin-configurable function allows for easy definition of action
  normalization parameters within Gin files. It sets up a normalizer that
  scales actions between a native range and a normalized range (typically
  [-1, 1] for RL agents).

  Args:
    min_native_value (float): The minimum value in the native (physical)
      setpoint range (e.g., minimum temperature).
    max_native_value (float): The maximum value in the native setpoint range.
    min_normalized_value (float): The minimum value of the normalized action
      range expected from/to the agent. Defaults to -1.0.
    max_normalized_value (float): The maximum value of the normalized action
      range. Defaults to 1.0.

  Returns:
    bounded_action_normalizer.BoundedActionNormalizer: An instance of
    `BoundedActionNormalizer` configured with the specified parameters.

  Example (in a .gin file):
    ```gin
    ActionConfig.action_normalizers = {
      'supply_temp_setpoint': @set_action_normalization_constants(
          min_native_value=288.0, # Kelvin, e.g., 15C
          max_native_value=300.0  # Kelvin, e.g., 27C
      )
    }
    ```
  """
  return bounded_action_normalizer.BoundedActionNormalizer(
      min_native_value=min_native_value,
      max_native_value=max_native_value,
      min_normalized_value=min_normalized_value,
      max_normalized_value=max_normalized_value,
  )
