"""Defines an occupancy model that returns the average number of occupants.

Occupancy refers to the average number of people in a zone within a specified
period of time. Concrete classes can either simulate the occupancy or
estimate the occupancy from Calendar or motion sensors in the buildings.

The occupancy signal is an input to the agent's reward function.

    Copyright 2022 Google LLC

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import abc

import pandas as pd


class BaseOccupancy(metaclass=abc.ABCMeta):
  """Abstract base class for providing zone occupancy information.

  This class defines an interface for models that estimate or report the
  average number of occupants within a specific zone of a building over a
  defined time period. Concrete implementations might derive this information
  from various sources, such as:
  - Building simulation models.
  - Occupancy schedules (e.g., from calendar systems).
  - Real-time sensor data (e.g., motion sensors, CO2 sensors, Wi-Fi connections).

  The occupancy data provided by this class is often a critical input for
  calculating occupant comfort, determining ventilation needs, or influencing
  the reward signal in reinforcement learning-based building control systems.
  """

  @abc.abstractmethod
  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Calculates the average number of occupants in a zone over an interval.

    Args:
      zone_id: The unique identifier for the building zone (e.g., "Office_101",
        "ConferenceRoom_A").
      start_time: A `pandas.Timestamp` representing the beginning of the time
        interval (local time, timezone-aware).
      end_time: A `pandas.Timestamp` representing the end of the time interval
        (local time, timezone-aware).

    Returns:
      The average number of people estimated or known to be in the specified
      zone during the given time interval. This is a float value to account
      for averaging over time.

    Raises:
      ValueError: If the provided `zone_id` is not recognized or if occupancy
        data is unavailable for the given zone or time period.
    """
    pass
