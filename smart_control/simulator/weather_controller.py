"""Provides weather data and conditions for the building simulation.

This module defines controllers that supply ambient temperature and convective
heat transfer coefficients to the simulator. It includes:
- `BaseWeatherController`: An abstract base class for weather models.
- `WeatherController`: A model generating sinusoidal daily temperature profiles
  with options for specifying different temperatures on special days.
- `ReplayWeatherController`: A model that replays historical weather data from
  a CSV file, interpolating temperatures for requested timestamps.
"""

import abc
import math
from typing import Final, Mapping, Optional, Sequence, Tuple

import gin
import numpy as np
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils

TemperatureBounds = Tuple[float, float]
"""Type alias for a tuple representing (low_temp_k, high_temp_k)."""

_SECONDS_IN_A_DAY: Final[float] = 24.0 * 3600.0
_DAYS_IN_A_YEAR: Final[int] = 365 # Used for modulo arithmetic on day of year
_MIN_RADIANS_FOR_SINE_MODEL: Final[float] = -math.pi / 2.0 # Start of sine wave
_MAX_RADIANS_FOR_SINE_MODEL: Final[float] = 3.0 * math.pi / 2.0 # End of sine wave
_EPOCH_TIMESTAMP: Final[pd.Timestamp] = pd.Timestamp("1970-01-01", tz="UTC")


@gin.configurable
class BaseWeatherController(metaclass=abc.ABCMeta):
  """Abstract base class for weather controllers.

  Defines the interface for providing current ambient temperature and
  convection coefficients to the building simulator.
  """

  @abc.abstractmethod
  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Returns the ambient outdoor temperature (K) for the given timestamp.

    Args:
      timestamp (pd.Timestamp): The specific time for which to get the
        temperature.
    """

  @abc.abstractmethod
  def get_air_convection_coefficient(self, timestamp: pd.Timestamp) -> float:
    """Returns the air convection coefficient (W/m^2K) for the given timestamp.

    This coefficient is used in calculating heat transfer between building
    surfaces and the outdoor environment.

    Args:
      timestamp (pd.Timestamp): The specific time for which to get the
        convection coefficient.
    """


@gin.configurable
class WeatherController(BaseWeatherController):
  """Simulates daily weather with sinusoidal temperature variation.

  This controller models outdoor temperature as a daily sine wave, peaking at
  noon and reaching its low at midnight. It allows for default high/low
  temperatures and overrides for specific days of the year.

  Attributes:
    default_low_temp_k (float): Default low temperature in Kelvin, typically
      occurring at midnight.
    default_high_temp_k (float): Default high temperature in Kelvin, typically
      occurring at noon.
    special_days (Mapping[int, TemperatureBounds]): A map where keys are days
      of the year (1-365) and values are `TemperatureBounds` (low_temp_k,
      high_temp_k) tuples to override default temperatures for those specific days.
    convection_coefficient_w_m2k (float): The air convection coefficient
      (W/m^2K) used for calculating heat loss/gain from building surfaces.
      This is currently constant in this model.
  """

  def __init__(
      self,
      default_low_temp_k: float,
      default_high_temp_k: float,
      special_days: Optional[Mapping[int, TemperatureBounds]] = None,
      convection_coefficient_w_m2k: float = 12.0,
  ):
    """Initializes the WeatherController.

    Args:
      default_low_temp_k (float): Default daily low temperature (K).
      default_high_temp_k (float): Default daily high temperature (K).
      special_days (Optional[Mapping[int, TemperatureBounds]]): Mapping for
        overriding temperatures on specific days of the year (1-365).
        Example: `{1: (270.0, 280.0)}` for January 1st.
      convection_coefficient_w_m2k (float): Convective heat transfer
        coefficient (W/m^2K).

    Raises:
      ValueError: If `default_low_temp_k` > `default_high_temp_k`, or if any
        special day has low_temp > high_temp.
    """
    if default_low_temp_k > default_high_temp_k:
      raise ValueError(
          "Default low temperature cannot be greater than default high temperature."
      )
    self.default_low_temp_k = default_low_temp_k
    self.default_high_temp_k = default_high_temp_k
    self.special_days = special_days if special_days else {}
    self.convection_coefficient_w_m2k = convection_coefficient_w_m2k

    for day_of_year, (low_k, high_k) in self.special_days.items():
      if not (1 <= day_of_year <= _DAYS_IN_A_YEAR):
          raise ValueError(f"Special day {day_of_year} out of range (1-365).")
      if low_k > high_k:
        raise ValueError(
            f"Low temperature ({low_k}K) cannot be greater than high "
            f"temperature ({high_k}K) for special day: {day_of_year}."
        )

  def _seconds_to_radians(self, seconds_into_day: float) -> float:
    """Converts seconds past midnight to radians for the sinusoidal model.

    The sine wave for temperature is modeled from -pi/2 (midnight, low) to
    +pi/2 (noon, high) and back to 3pi/2 (next midnight, low).

    Args:
      seconds_into_day (float): Number of seconds elapsed since midnight.

    Returns:
      float: The corresponding angle in radians for the sine function.
    """
    fraction_of_day = seconds_into_day / _SECONDS_IN_A_DAY
    return (
        fraction_of_day * (_MAX_RADIANS_FOR_SINE_MODEL - _MIN_RADIANS_FOR_SINE_MODEL) +
        _MIN_RADIANS_FOR_SINE_MODEL
    )

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Calculates ambient temperature (K) using a sinusoidal daily profile.

    Args:
      timestamp (pd.Timestamp): The specific time for which to get the
        temperature.

    Returns:
      float: The calculated ambient temperature in Kelvin.
    """
    day_of_year = timestamp.dayofyear # 1-366 for leap years
    # Normalize to 1-365 for consistency with special_days mapping
    normalized_day_of_year = day_of_year if day_of_year <= 365 else 365


    # Determine today's and tomorrow's low/high temperatures
    today_low_k, today_high_k = self.special_days.get(
        normalized_day_of_year,
        (self.default_low_temp_k, self.default_high_temp_k)
    )

    # For interpolation, need tomorrow's low if current time is past noon.
    # Handle year wrap-around for tomorrow's day of year.
    next_day_of_year = (timestamp + pd.Timedelta(days=1)).dayofyear
    normalized_next_day_of_year = next_day_of_year if next_day_of_year <= 365 else 365
    tomorrow_low_k, _ = self.special_days.get(
        normalized_next_day_of_year,
        (self.default_low_temp_k, self.default_high_temp_k) # Only need low
    )

    # The sine wave uses today's low and high if before noon,
    # or today's high and tomorrow's low if after noon.
    current_day_high_k = today_high_k
    current_day_low_k = today_low_k if timestamp.hour < 12 else tomorrow_low_k

    seconds_since_midnight = (
        timestamp - timestamp.normalize()
    ).total_seconds()
    angle_rad = self._seconds_to_radians(seconds_since_midnight)

    # Sinusoidal interpolation: sin(rad) goes from -1 (midnight) to +1 (noon)
    # So, (sin(rad) + 1) / 2 scales it to 0 (midnight) to 1 (noon).
    temperature_k = (
        0.5 * (math.sin(angle_rad) + 1.0) *
        (current_day_high_k - current_day_low_k) +
        current_day_low_k
    )
    return temperature_k

  def get_air_convection_coefficient(
      self, timestamp: pd.Timestamp # pylint: disable=unused-argument
  ) -> float:
    """Returns the air convection coefficient (W/m^2K).

    Currently, this model uses a constant convection coefficient.

    Args:
      timestamp (pd.Timestamp): The current time (unused in this implementation).

    Returns:
      float: The configured constant convection coefficient.
    """
    return self.convection_coefficient_w_m2k


def get_replay_temperatures(
    observation_responses: Sequence[smart_control_building_pb2.ObservationResponse],
) -> Mapping[str, float]:
  """Extracts historical outside air temperatures from observation data.

  This function processes a sequence of `ObservationResponse` protobuf messages
  and extracts the 'outside_air_temperature_sensor' readings, creating a
  mapping from timestamp strings to temperature values.

  Args:
    observation_responses (Sequence[ObservationResponse]): A sequence of
      observation responses, each potentially containing an outside air
      temperature reading.

  Returns:
    Mapping[str, float]: A dictionary where keys are ISO format timestamp
    strings (UTC) and values are the corresponding outside air temperatures
    in Kelvin. Returns -1.0 for a timestamp if temperature is not found.
  """

  def get_outside_air_temp_from_response(
      response: smart_control_building_pb2.ObservationResponse
  ) -> float:
    """Helper to find outside air temp in a single ObservationResponse."""
    for r_field in response.single_observation_responses:
      if (r_field.single_observation_request.measurement_name ==
          "outside_air_temperature_sensor"):
        return r_field.continuous_value
    return -1.0 # Sentinel for not found

  temps_map: dict[str, float] = {}
  for response_item in observation_responses:
    temp_k = get_outside_air_temp_from_response(response_item)
    timestamp_pd = conversion_utils.proto_to_pandas_timestamp(response_item.timestamp)
    temps_map[timestamp_pd.isoformat()] = temp_k
  return temps_map


@gin.configurable
class ReplayWeatherController(BaseWeatherController):
  """Provides weather data by replaying and interpolating historical records.

  This controller loads weather data from a CSV file (expected to have 'Time'
  and 'TempF' columns). It then provides interpolated temperature values for
  any requested timestamp within the range of the historical data.

  Attributes:
    _weather_data (pd.DataFrame): DataFrame holding the historical weather
      data, indexed by seconds since epoch for efficient interpolation.
    convection_coefficient_w_m2k (float): Air convection coefficient (W/m^2K).
      This is constant in this model.
  """

  def __init__(
      self,
      local_weather_csv_path: str,
      convection_coefficient_w_m2k: float = 12.0,
  ):
    """Initializes the ReplayWeatherController.

    Args:
      local_weather_csv_path (str): Path to the CSV file containing historical
        weather data. The CSV must include 'Time' (parsable by
        `pd.Timestamp`) and 'TempF' (temperature in Fahrenheit) columns.
      convection_coefficient_w_m2k (float): Constant air convection
        coefficient (W/m^2K).
    """
    try:
      self._weather_data = pd.read_csv(local_weather_csv_path)
    except FileNotFoundError:
      logging.error("Weather data file not found at: %s", local_weather_csv_path)
      raise
    if "Time" not in self._weather_data.columns or \
       "TempF" not in self._weather_data.columns:
        raise ValueError("Weather CSV must contain 'Time' and 'TempF' columns.")

    # Convert 'Time' column to timezone-aware UTC Timestamps
    self._weather_data["Time"] = pd.to_datetime(
        self._weather_data["Time"]
    ).dt.tz_localize("UTC")
    # Create an index based on seconds since epoch for interpolation
    self._weather_data.index = (
        self._weather_data["Time"] - _EPOCH_TIMESTAMP
    ).dt.total_seconds()
    self._weather_data.sort_index(inplace=True) # Ensure index is sorted

    self.convection_coefficient_w_m2k = convection_coefficient_w_m2k

  def get_current_temp(self, timestamp: pd.Timestamp) -> float:
    """Returns interpolated outdoor temperature (K) for the given timestamp.

    Args:
      timestamp (pd.Timestamp): The specific time for which to get the
        temperature. Must be timezone-aware or will be localized to UTC.

    Returns:
      float: Interpolated ambient temperature in Kelvin.

    Raises:
      ValueError: If the requested `timestamp` is outside the range of the
        loaded historical weather data.
    """
    if timestamp.tzinfo is None: # Ensure timestamp is timezone-aware
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC") # Convert to UTC for comparison

    min_time_data = self._weather_data["Time"].min()
    max_time_data = self._weather_data["Time"].max()

    if not (min_time_data <= timestamp <= max_time_data):
      raise ValueError(
          f"Requested timestamp {timestamp} is outside the available weather "
          f"data range ({min_time_data} to {max_time_data})."
      )

    target_seconds_since_epoch = (timestamp - _EPOCH_TIMESTAMP).total_seconds()
    # Interpolate temperature in Fahrenheit
    interp_temp_f = np.interp(
        target_seconds_since_epoch,
        self._weather_data.index, # Seconds since epoch
        self._weather_data["TempF"]
    )
    return conversion_utils.fahrenheit_to_kelvin(interp_temp_f)

  def get_air_convection_coefficient(
      self, timestamp: pd.Timestamp # pylint: disable=unused-argument
  ) -> float:
    """Returns the air convection coefficient (W/m^2K).

    Currently, this model uses a constant convection coefficient.

    Args:
      timestamp (pd.Timestamp): The current time (unused).

    Returns:
      float: The configured constant convection coefficient.
    """
    return self.convection_coefficient_w_m2k
