"""General-purpose conversion utilities for the Smart Control project.

This module provides helper functions for various data type and unit
conversions commonly needed across the smart building simulation and control
framework. These include:
- Timestamp conversions between Pandas and Protobuf formats.
- Workday and holiday determination.
- Zone identifier conversions.
- Normalization of time-based features like day of week and hour of day.
- Temperature unit conversions (Kelvin to Fahrenheit and vice-versa).
- Calculation of energy consumption from power rates.
"""

import collections
import datetime
import enum
import functools
import re
import types # For types.MappingProxyType
from typing import Mapping, Tuple

from google.protobuf import timestamp_pb2
import holidays # For US holidays
import numpy as np
import pandas as pd

from smart_control.proto import smart_control_reward_pb2

_US_COUNTRY_CODE = "US"
_SECONDS_PER_DAY: float = 24.0 * 3600.0
_WATT_SECONDS_PER_KWH: float = 3600.0 * 1000.0 # 1 kWh = 1000 Wh = 1000 * 3600 Ws
_DAYS_PER_WEEK: float = 7.0


def pandas_to_proto_timestamp(
    pandas_ts: pd.Timestamp,
) -> timestamp_pb2.Timestamp:
  """Converts a Pandas Timestamp object to a Google Protobuf Timestamp.

  Args:
    pandas_ts (pd.Timestamp): The Pandas Timestamp to convert. It should be
      timezone-aware (e.g., localized to UTC) for accurate conversion of
      `timestamp()` method.

  Returns:
    timestamp_pb2.Timestamp: The equivalent Protobuf Timestamp.
  """
  proto_ts = timestamp_pb2.Timestamp()
  # pd.Timestamp.timestamp() returns POSIX timestamp (seconds since epoch UTC)
  proto_ts.seconds = int(pandas_ts.timestamp())
  # Pandas microsecond and nanosecond attributes are for the fractional part
  proto_ts.nanos = pandas_ts.microsecond * 1000 + pandas_ts.nanosecond
  return proto_ts


def proto_to_pandas_timestamp(
    proto_ts: timestamp_pb2.Timestamp,
) -> pd.Timestamp:
  """Converts a Google Protobuf Timestamp to a Pandas Timestamp object.

  The resulting Pandas Timestamp will be localized to UTC.

  Args:
    proto_ts (timestamp_pb2.Timestamp): The Protobuf Timestamp to convert.

  Returns:
    pd.Timestamp: The equivalent Pandas Timestamp, localized to UTC.
  """
  return pd.Timestamp(
      proto_ts.seconds, unit="s", tz="UTC"
  ) + pd.Timedelta(proto_ts.nanos, unit="ns")


@functools.lru_cache(maxsize=1) # Cache the result as holidays don't change often
def _get_us_holidays() -> types.MappingProxyType:
  """Returns a read-only mapping of US holidays for the current year.

  Uses the `holidays` library. The result is cached for efficiency.

  Returns:
    types.MappingProxyType: A mapping where keys are `datetime.date` objects
    representing US holidays and values are the holiday names (str).
  """
  # Using a specific year or a range might be more robust if needed across years.
  # For now, defaults to current year of `holidays` library.
  return types.MappingProxyType(holidays.country_holidays(_US_COUNTRY_CODE))


def is_work_day(timestamp: pd.Timestamp) -> bool:
  """Determines if a given timestamp falls on a workday (Mon-Fri, not a US holiday).

  Args:
    timestamp (pd.Timestamp): The timestamp to check.

  Returns:
    bool: True if the timestamp is a workday, False otherwise.
  """
  date_to_check = timestamp.date()
  # Monday is 0, Sunday is 6 for weekday()
  is_weekday = timestamp.weekday() < 5 # Monday to Friday
  is_holiday = date_to_check in _get_us_holidays()
  return is_weekday and not is_holiday


def zone_coordinates_to_id(coordinates: Tuple[int, int]) -> str:
  """Converts 2D zone coordinates to a standardized string ID.

  Args:
    coordinates (Tuple[int, int]): A tuple (row, col) representing the zone's
      position in a grid.

  Returns:
    str: A string identifier for the zone, e.g., "zone_id_(0, 1)".
  """
  return f"zone_id_{str(coordinates)}"


def floor_plan_based_zone_identifier_to_id(room_identifier: str) -> str:
  """Converts a room identifier (often from a floor plan) to a zone ID.

  Assumes room identifiers might be like "room_A1" or "room_101". This function
  prefixes it with "zone_id_" and removes "room_".

  Args:
    room_identifier (str): The original identifier for the room/zone.

  Returns:
    str: A standardized zone ID string, e.g., "zone_id_A1".
  """
  return f"zone_id_{room_identifier.replace('room_', '')}"


def zone_id_to_coordinates(zone_id_str: str) -> Tuple[int, int]:
  """Converts a standardized zone string ID back to 2D coordinates.

  This is the inverse of `zone_coordinates_to_id`.

  Args:
    zone_id_str (str): The zone ID string (e.g., "zone_id_(0, 1)").

  Returns:
    Tuple[int, int]: The (row, col) coordinates.

  Raises:
    ValueError: If the `zone_id_str` is not in the expected format.
  """
  pattern = r"^zone_id_\((\d+), (\d+)\)$" # Matches "zone_id_(row, col)"
  match = re.fullmatch(pattern, zone_id_str)
  if match:
    return (int(match.group(1)), int(match.group(2)))
  raise ValueError(
      f"Could not convert zone ID '{zone_id_str}' to coordinates. "
      "Expected format: 'zone_id_(row, col)'."
  )


def normalize_dow(day_of_week_int: int) -> float:
  """Normalizes day of the week (0=Mon, 6=Sun) to the range [-1.0, 1.0].

  Args:
    day_of_week_int (int): Day of the week, where Monday is 0 and Sunday is 6.

  Returns:
    float: Normalized day of the week.

  Raises:
    AssertionError: If `day_of_week_int` is not in the range [0, 6].
  """
  assert 0 <= day_of_week_int <= 6, "Day of week must be between 0 and 6."
  # Maps 0 (Mon) to -1.0, 3 (Thu) to 0.0, 6 (Sun) to 1.0
  return (float(day_of_week_int) - 3.0) / 3.0


def normalize_hod(hour_of_day_int: int) -> float:
  """Normalizes hour of the day (0-23) to the range [-1.0, 1.0].

  Args:
    hour_of_day_int (int): Hour of the day, from 0 to 23.

  Returns:
    float: Normalized hour of the day.

  Raises:
    AssertionError: If `hour_of_day_int` is not in the range [0, 23].
  """
  assert 0 <= hour_of_day_int <= 23, "Hour of day must be between 0 and 23."
  # Maps 0 to -1.0, 11.5 to 0.0, 23 to 1.0
  return (float(hour_of_day_int) - 11.5) / 11.5


class TimeIntervalEnum(enum.Enum):
  """Enumerates types of time intervals for cyclical feature conversion."""
  DAY_OF_WEEK = 1
  HOUR_OF_DAY = 2 # TODO(b/260300338): Consider renaming to TIME_OF_DAY for clarity.


def get_radian_time(
    timestamp: pd.Timestamp, time_interval_type: TimeIntervalEnum
) -> float:
  """Converts a timestamp to a radian value representing its position in a cycle.

  The output ranges from 0 to 2*pi, representing a full cycle of the specified
  `time_interval_type` (e.g., a week or a day).

  Args:
    timestamp (pd.Timestamp): The timestamp to convert. Assumed to be in the
      local timezone relevant to the building's operation schedule.
    time_interval_type (TimeIntervalEnum): The type of cycle to consider
      (e.g., DAY_OF_WEEK, HOUR_OF_DAY).

  Returns:
    float: The radian value (0 to 2*pi) representing the timestamp's position
    within the specified cycle.

  Raises:
    ValueError: If an unsupported `time_interval_type` is provided.
  """
  if time_interval_type == TimeIntervalEnum.DAY_OF_WEEK:
    # Monday is 0, Sunday is 6 for weekday()
    # Add seconds/microseconds for finer granularity within the day.
    day_fraction = timestamp.weekday() + (
        (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second +
         timestamp.microsecond / 1e6) / _SECONDS_PER_DAY
    )
    interval_fraction = day_fraction / _DAYS_PER_WEEK
  elif time_interval_type == TimeIntervalEnum.HOUR_OF_DAY:
    seconds_into_day = (
        timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second +
        timestamp.microsecond / 1e6
    )
    interval_fraction = seconds_into_day / _SECONDS_PER_DAY
  else:
    raise ValueError(f"Unsupported time interval type: {time_interval_type}")
  return 2.0 * np.pi * interval_fraction


def kelvin_to_fahrenheit(temp_kelvin: float) -> float:
  """Converts temperature from Kelvin to Fahrenheit.

  Args:
    temp_kelvin (float): Temperature in Kelvin.

  Returns:
    float: Temperature in Fahrenheit.

  Raises:
    ValueError: If `temp_kelvin` is at or below absolute zero (0 K).
  """
  if temp_kelvin <= 0.0:
    raise ValueError("Temperature must be greater than absolute zero (0 K).")
  temp_celsius = temp_kelvin - 273.15
  return (temp_celsius * 9.0 / 5.0) + 32.0


def fahrenheit_to_kelvin(temp_fahrenheit: float) -> float:
  """Converts temperature from Fahrenheit to Kelvin.

  Args:
    temp_fahrenheit (float): Temperature in Fahrenheit.

  Returns:
    float: Temperature in Kelvin.

  Raises:
    ValueError: If `temp_fahrenheit` is at or below absolute zero (-459.67 °F).
  """
  # Absolute zero in Fahrenheit is -459.67 °F.
  if temp_fahrenheit <= -459.67:
    raise ValueError(
        "Temperature must be greater than absolute zero (-459.67 °F)."
    )
  temp_celsius = (temp_fahrenheit - 32.0) * 5.0 / 9.0
  return temp_celsius + 273.15


def get_reward_info_energy_use(
    reward_info: smart_control_reward_pb2.RewardInfo,
) -> Mapping[str, float]:
  """Calculates energy consumption (kWh) for components from `RewardInfo`.

  Extracts power rates from `reward_info` for air handler (blower, AC) and
  boiler (gas heating, pump electricity), multiplies by the time duration of
  the `reward_info` interval, and converts to kilowatt-hours (kWh).

  Args:
    reward_info (smart_control_reward_pb2.RewardInfo): A protobuf message
      containing energy consumption rates and interval start/end times.

  Returns:
    Mapping[str, float]: A dictionary where keys are component energy use
    identifiers (e.g., 'air_handler_blower_electricity') and values are the
    energy consumed in kWh during the interval.
  """
  start_pd_ts = proto_to_pandas_timestamp(reward_info.start_timestamp)
  end_pd_ts = proto_to_pandas_timestamp(reward_info.end_timestamp)
  duration_seconds = (end_pd_ts - start_pd_ts).total_seconds()

  energy_use_kwh = collections.defaultdict(float)
  watt_seconds_to_kwh_factor = 1.0 / _WATT_SECONDS_PER_KWH

  for ah_id, ah_info in reward_info.air_handler_reward_infos.items():
    energy_use_kwh["air_handler_blower_electricity"] += (
        ah_info.blower_electrical_energy_rate * duration_seconds *
        watt_seconds_to_kwh_factor
    )
    # AC energy can be positive (heating) or negative (cooling),
    # absolute value represents energy consumed.
    energy_use_kwh["air_handler_air_conditioning"] += (
        abs(ah_info.air_conditioning_electrical_energy_rate) * duration_seconds *
        watt_seconds_to_kwh_factor
    )

  for boiler_id, boiler_info in reward_info.boiler_reward_infos.items():
    energy_use_kwh["boiler_natural_gas_heating_energy"] += (
        boiler_info.natural_gas_heating_energy_rate * duration_seconds *
        watt_seconds_to_kwh_factor
    )
    energy_use_kwh["boiler_pump_electrical_energy"] += (
        boiler_info.pump_electrical_energy_rate * duration_seconds *
        watt_seconds_to_kwh_factor
    )

  return energy_use_kwh
