"""Abstract base class for building occupancy models.

This module defines `BaseOccupancy`, an abstract interface for models that
provide information about the number of occupants within different zones of a
building over time. Occupancy data is a critical input for demand-driven
building control strategies and can significantly influence the reward signal
for a reinforcement learning agent.

Implementing classes might derive occupancy from various sources, such as:
- Simulation based on predefined schedules.
- Real-time sensor data (e.g., motion sensors, CO2 levels).
- Calendar information indicating meetings or events.

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc

import pandas as pd


class BaseOccupancy(metaclass=abc.ABCMeta):
  """Abstract interface for providing zone occupancy information.

  This class defines the method that concrete occupancy models must implement
  to report the average number of occupants in a specific zone over a given
  time interval. This information is then used by the RL agent or its
  reward function.

  Conceptual Example:
    A simple schedule-based occupancy model:

    ```python
    class ScheduledOccupancy(BaseOccupancy):
        def __init__(self, occupancy_schedules_file):
            # Schedules might map (zone_id, day_of_week, hour) to counts
            self._schedules = self._load_schedules(occupancy_schedules_file)

        def _load_schedules(self, file_path):
            # Load and parse schedule data
            pass # ... implementation ...

        def average_zone_occupancy(self, zone_id, start_time, end_time):
            if zone_id not in self._schedules:
                raise ValueError(f"Zone ID {zone_id} not found in schedules.")
            # For simplicity, this example might just use the occupancy
            # at the start_time or average over the interval based on schedule.
            # A more complex model would integrate over the interval.
            occupancy_count = self._schedules.get(
                zone_id, {}).get(start_time.dayofweek, {}).get(start_time.hour, 0)
            return float(occupancy_count)
    ```
  """

  @abc.abstractmethod
  def average_zone_occupancy(
      self, zone_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> float:
    """Calculates the average occupancy for a given zone and time interval.

    Implementations should determine the average number of people present in
    the specified `zone_id` between `start_time` (inclusive) and `end_time`
    (exclusive or inclusive, depending on convention adopted by the
    implementation).

    Args:
      zone_id (str): The unique identifier for the building zone.
      start_time (pd.Timestamp): The local start time of the interval,
        inclusive, with time zone information.
      end_time (pd.Timestamp): The local end time of the interval, typically
        exclusive, with time zone information.

    Returns:
      float: The average number of occupants in the zone during the specified
      interval. This can be a fractional value if the underlying model
      represents probabilities or averages over finer granularities.

    Raises:
      ValueError: If the specified `zone_id` is not recognized by the model.
    """
    pass
