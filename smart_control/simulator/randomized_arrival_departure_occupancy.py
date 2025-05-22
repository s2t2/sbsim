"""A stochastic occupancy model based on randomized arrival and departure times.

This module provides a more dynamic occupancy simulation compared to fixed
schedules. It models individual occupants (`ZoneOccupant`) who arrive and depart
within specified time windows, with probabilities calculated to make arrivals/
departures more likely around the midpoint of these windows. The main class,
`RandomizedArrivalDepartureOccupancy`, manages collections of these occupants
for different zones to provide an overall zone occupancy count.
"""

import datetime
import enum
from typing import Optional, Union, Dict, List # Added Dict, List

import gin
import numpy as np
import pandas as pd

from smart_control.models.base_occupancy import BaseOccupancy
from smart_control.utils import conversion_utils


class OccupancyStateEnum(enum.Enum):
  """Enumerates the possible states of an individual occupant."""
  AWAY = 1  # Occupant is not in the zone/building.
  WORK = 2  # Occupant is present in the zone/building.


class ZoneOccupant:
  """Simulates a single occupant with randomized arrival and departure behavior.

  An occupant has defined time windows for their expected arrival and departure.
  Within these windows, the probability of arriving or departing at any given
  simulation step is calculated to make the event more likely towards the
  middle of the window. This creates a stochastic pattern of presence.
  """

  def __init__(
      self,
      earliest_expected_arrival_hour: int,
      latest_expected_arrival_hour: int,
      earliest_expected_departure_hour: int,
      latest_expected_departure_hour: int,
      step_size: pd.Timedelta,
      random_state: np.random.RandomState,
      time_zone: Union[datetime.tzinfo, str] = 'UTC',
  ):
    """Initializes a ZoneOccupant.

    Args:
      earliest_expected_arrival_hour: The earliest hour (0-23, local time)
        the occupant might arrive.
      latest_expected_arrival_hour: The latest hour (0-23, local time)
        the occupant might arrive. Must be > earliest_expected_arrival_hour.
      earliest_expected_departure_hour: The earliest hour (0-23, local time)
        the occupant might depart. Must be > latest_expected_arrival_hour.
      latest_expected_departure_hour: The latest hour (0-23, local time)
        the occupant might depart. Must be > earliest_expected_departure_hour.
      step_size: A `pd.Timedelta` representing the duration of each simulation
        step. Used to calculate per-step event probabilities.
      random_state: A `np.random.RandomState` instance for generating random
        numbers for arrival/departure events, ensuring reproducibility if seeded.
      time_zone: The timezone (e.g., 'America/Los_Angeles', 'UTC') to which
        timestamps should be localized before checking against hourly windows.
        Defaults to 'UTC'.

    Raises:
      AssertionError: If the hour windows are not logically ordered.
    """
    assert (
        earliest_expected_arrival_hour < latest_expected_arrival_hour
        < earliest_expected_departure_hour < latest_expected_departure_hour
    ), "Hour windows for arrival/departure are not correctly ordered."

    self._earliest_expected_arrival_hour = earliest_expected_arrival_hour
    self._latest_expected_arrival_hour = latest_expected_arrival_hour
    self._earliest_expected_departure_hour = earliest_expected_departure_hour
    self._latest_expected_departure_hour = latest_expected_departure_hour
    self._step_size = step_size
    self._occupancy_state = OccupancyStateEnum.AWAY # Start as AWAY
    # Probability of arriving in a single step within the arrival window
    self._p_arrival = self._get_event_probability(
        earliest_expected_arrival_hour, latest_expected_arrival_hour
    )
    # Probability of departing in a single step within the departure window
    self._p_departure = self._get_event_probability(
        earliest_expected_departure_hour, latest_expected_departure_hour
    )
    self._random_state = random_state
    self._time_zone = time_zone

  def _to_local_time(self, timestamp: pd.Timestamp) -> pd.Timestamp:
    """Converts a pandas Timestamp to the occupant's configured local timezone.

    If the input timestamp is naive, it's assumed to be UTC before conversion.
    If timezone-aware, it's directly converted.

    Args:
      timestamp: The `pd.Timestamp` to convert.

    Returns:
      The timestamp localized to `self._time_zone`.
    """
    if timestamp.tzinfo is None:
      return timestamp.tz_localize('UTC').tz_convert(self._time_zone)
    else:
      return timestamp.tz_convert(self._time_zone)

  def _get_event_probability(self, start_hour: int, end_hour: int) -> float:
    """Calculates per-step probability for an event within a time window.

    The probability is set such that the event (arrival or departure) is
    expected to occur, on average, at the midpoint of the window
    [`start_hour`, `end_hour`]. This uses the property of a geometric
    distribution where E[X] = 1/p for the number of trials to first success.

    Args:
      start_hour: The starting hour of the window (inclusive).
      end_hour: The ending hour of the window (exclusive, effectively making
        the window `end_hour - start_hour` long).

    Returns:
      The per-step probability (float) of the event occurring.
    """
    assert start_hour < end_hour, "Start hour must be less than end hour for event window."
    window_duration = pd.Timedelta(hours=(end_hour - start_hour))
    # Number of simulation steps until the midpoint of the window
    num_steps_to_midpoint = (window_duration / self._step_size) / 2.0
    if num_steps_to_midpoint <= 0: # Avoid division by zero or negative probability
        return 1.0 # Event happens immediately if window is too small or step size too large
    return 1.0 / num_steps_to_midpoint

  def _occupant_arrived(self, timestamp: pd.Timestamp) -> bool:
    """Determines if the occupant arrives at the given timestamp.

    Arrival can only happen if the current local hour is within the defined
    arrival window. Within this window, arrival occurs probabilistically based
    on `self._p_arrival`.

    Args:
      timestamp: The current simulation timestamp.

    Returns:
      True if the occupant arrives at this step, False otherwise.
    """
    local_timestamp = self._to_local_time(timestamp)
    # TODO(sipple): Consider effects when time crosses DST. (Original comment)
    # Current logic uses local_timestamp.hour, which should handle DST correctly
    # as long as pd.Timestamp localization/conversion is correct.
    if not (self._earliest_expected_arrival_hour <= local_timestamp.hour < self._latest_expected_arrival_hour):
      return False
    return self._random_state.rand() < self._p_arrival

  def _occupant_departed(self, timestamp: pd.Timestamp) -> bool:
    """Determines if the occupant departs at the given timestamp.

    Departure can only happen if the current local hour is at or after the
    earliest departure hour. Within the valid departure window (implicitly up
    to day end or `latest_expected_departure_hour`), departure occurs
    probabilistically based on `self._p_departure`.

    Args:
      timestamp: The current simulation timestamp.

    Returns:
      True if the occupant departs at this step, False otherwise.
    """
    local_timestamp = self._to_local_time(timestamp)
    # Departure can happen from earliest_expected_departure_hour up to latest_expected_departure_hour
    if not (self._earliest_expected_departure_hour <= local_timestamp.hour < self._latest_expected_departure_hour):
      return False
    return self._random_state.rand() < self._p_departure

  def peek(self, current_time: pd.Timestamp) -> OccupancyStateEnum:
    """Updates and returns the occupant's state (WORK or AWAY) for `current_time`.

    This method checks if the occupant should change their state based on work
    days, arrival/departure windows, and probabilistic checks.

    Args:
      current_time: The current simulation timestamp to evaluate occupancy for.

    Returns:
      The `OccupancyStateEnum` (AWAY or WORK) for the occupant at `current_time`.
    """
    local_timestamp = self._to_local_time(current_time)
    # Create a Timestamp for the day part only to check if it's a workday
    day_only_timestamp = pd.Timestamp(year=local_timestamp.year, month=local_timestamp.month, day=local_timestamp.day)

    if not conversion_utils.is_work_day(day_only_timestamp):
      self._occupancy_state = OccupancyStateEnum.AWAY
    elif self._occupancy_state == OccupancyStateEnum.AWAY:
      if self._occupant_arrived(current_time):
        self._occupancy_state = OccupancyStateEnum.WORK
    else:  # Occupant is currently WORK
      if self._occupant_departed(current_time):
        self._occupancy_state = OccupancyStateEnum.AWAY
    return self._occupancy_state


@gin.configurable
class RandomizedArrivalDepartureOccupancy(BaseOccupancy):
  """Simulates zone occupancy using multiple `ZoneOccupant` instances.

  This class implements the `BaseOccupancy` interface. For each zone, it manages
  a fixed number of `ZoneOccupant` objects, each with its own stochastic
  arrival and departure pattern. The overall occupancy of a zone at a given time
  is the sum of occupants in the `WORK` state.
  """

  def __init__(
      self,
      zone_assignment: int,
      earliest_expected_arrival_hour: int,
      latest_expected_arrival_hour: int,
      earliest_expected_departure_hour: int,
      latest_expected_departure_hour: int,
      time_step_sec: int,
      seed: Optional[int] = 17321,
      time_zone: str = 'UTC',
  ):
    """Initializes the RandomizedArrivalDepartureOccupancy model.

    Args:
      zone_assignment: The number of individual `ZoneOccupant` instances to
        simulate for each zone. This effectively sets the maximum number of
        occupants per zone if all were to arrive.
      earliest_expected_arrival_hour: Default earliest arrival hour (0-23) for
        each simulated occupant.
      latest_expected_arrival_hour: Default latest arrival hour (0-23) for
        each simulated occupant.
      earliest_expected_departure_hour: Default earliest departure hour (0-23)
        for each simulated occupant.
      latest_expected_departure_hour: Default latest departure hour (0-23) for
        each simulated occupant.
      time_step_sec: The duration of each simulation time step in seconds. This
        is used by `ZoneOccupant` to calculate per-step event probabilities.
      seed: An optional integer seed for the `np.random.RandomState` instance
        shared by all `ZoneOccupant`s, allowing for reproducible occupancy
        patterns. Defaults to 17321.
      time_zone: The default timezone string (e.g., 'America/Los_Angeles', 'UTC')
        for all `ZoneOccupant` instances. Defaults to 'UTC'.
    """
    self._zone_assignment: int = zone_assignment
    # Stores ZoneOccupant lists, keyed by zone_id string
    self._zone_occupants: Dict[str, List[ZoneOccupant]] = {}
    self._step_size: pd.Timedelta = pd.Timedelta(seconds=time_step_sec)
    self._earliest_expected_arrival_hour: int = earliest_expected_arrival_hour
    self._latest_expected_arrival_hour: int = latest_expected_arrival_hour
    self._earliest_expected_departure_hour: int = earliest_expected_departure_hour
    self._latest_expected_departure_hour: int = latest_expected_departure_hour
    self._random_state: np.random.RandomState = np.random.RandomState(seed)
    self._time_zone: str = time_zone

  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Calculates the number of occupants in a zone at a specific time.

    Note: Despite the name `average_zone_occupancy` and the `end_time`
    parameter (which is currently unused in the logic), this method returns the
    instantaneous number of occupants in the `WORK` state at `start_time`.
    If `zone_id` is encountered for the first time, `_zone_assignment` number
    of `ZoneOccupant` instances are created for it.

    Args:
      zone_id: The unique string identifier for the building zone.
      start_time: The `pandas.Timestamp` (expected to be timezone-aware or
        handled by `ZoneOccupant._to_local_time`) at which to determine
        occupancy.
      end_time: A `pandas.Timestamp` indicating the end of an interval.
        (Currently unused in this method's logic but part of BaseOccupancy interface).

    Returns:
      The number of occupants in the specified zone who are in the `WORK`
      state at `start_time`, returned as a float.
    """
    # Initialize occupants for the zone if not already done
    if zone_id not in self._zone_occupants:
      self._zone_occupants[zone_id] = [
          ZoneOccupant(
              earliest_expected_arrival_hour=self._earliest_expected_arrival_hour,
              latest_expected_arrival_hour=self._latest_expected_arrival_hour,
              earliest_expected_departure_hour=self._earliest_expected_departure_hour,
              latest_expected_departure_hour=self._latest_expected_departure_hour,
              step_size=self._step_size,
              random_state=self._random_state, # Shared RandomState for all occupants of this zone
              time_zone=self._time_zone,
          ) for _ in range(self._zone_assignment)
      ]

    current_occupant_count = 0.0
    for occupant in self._zone_occupants[zone_id]:
      if occupant.peek(start_time) == OccupancyStateEnum.WORK:
        current_occupant_count += 1.0
    return current_occupant_count
