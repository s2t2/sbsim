"""Abstract base class for building models in a reinforcement learning context.

This module defines the `BaseBuilding` class, an abstract interface for
representing and interacting with a building, whether it's a simulation or a
physical structure. It outlines the core functionalities required by the RL
environment, such as requesting observations, sending control actions, and
managing simulation time.

Implementing classes must provide concrete versions of all abstract methods
defined herein. This ensures a consistent API for the RL agent regardless of
the underlying building model.

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
from typing import Sequence

import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2


class BaseBuilding(metaclass=abc.ABCMeta):
  """Abstract interface for a controllable building model.

  This class defines the essential methods and properties that any building
  model (simulated or real) must implement to be compatible with the
  reinforcement learning environment. It handles state observation, action
  requests, time progression, and building configuration.

  Conceptual Example:
    A concrete implementation, `SimulatedOfficeBuilding`, might look like:

    ```python
    class SimulatedOfficeBuilding(BaseBuilding):
        def __init__(self, config_file):
            # Load building parameters, simulation engine, etc.
            self._simulation_time = pd.Timestamp("2023-01-01 00:00:00")
            self._hvac_setpoints = {} # Store current setpoints
            # ... other initialization ...

        @property
        def reward_info(self):
            # Gather data like energy use, comfort metrics from sim
            # and return as RewardInfo proto.
            pass # ... implementation ...

        def request_observations(self, observation_request):
            # Query internal simulation state based on request
            # and return as ObservationResponse proto.
            pass # ... implementation ...

        # ... other method implementations ...

        def reset(self):
            # Reset simulation to initial state.
            self._simulation_time = pd.Timestamp("2023-01-01 00:00:00")
            # ...
    ```
  """

  @property
  @abc.abstractmethod
  def reward_info(self) -> smart_control_reward_pb2.RewardInfo:
    """Retrieves data used to compute the instantaneous reward.

    This property should return a `RewardInfo` protobuf message containing
    all necessary information (e.g., energy consumption, comfort metrics)
    for the reward function to calculate the current step's reward.

    Returns:
      smart_control_reward_pb2.RewardInfo: A protobuf message populated with
      data relevant to reward calculation.
    """

  @abc.abstractmethod
  def request_observations(
      self, observation_request: smart_control_building_pb2.ObservationRequest
  ) -> smart_control_building_pb2.ObservationResponse:
    """Queries the building for its current state observations.

    Implementing classes should process the `observation_request`, gather the
    specified data points from the building model, and return them in an
    `ObservationResponse` protobuf message.

    Args:
      observation_request (smart_control_building_pb2.ObservationRequest):
        A protobuf message specifying which observations are requested.

    Returns:
      smart_control_building_pb2.ObservationResponse: A protobuf message
      containing the requested observations and their current values.
    """

  @abc.abstractmethod
  def request_observations_within_time_interval(
      self,
      observation_request: smart_control_building_pb2.ObservationRequest,
      start_timestamp: pd.Timestamp,
      end_timestamp: pd.Timestamp,
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Queries for a sequence of observations within a specified time range.

    This method is typically used for fetching historical data or for models
    that can provide observations over a span of time, not just the current
    step.

    Args:
      observation_request (smart_control_building_pb2.ObservationRequest):
        Specifies the types of observations requested. The `timestamp` field
        within this request might be ignored or used as a reference.
      start_timestamp (pd.Timestamp): The inclusive start of the time interval.
      end_timestamp (pd.Timestamp): The inclusive end of the time interval.

    Returns:
      Sequence[smart_control_building_pb2.ObservationResponse]: A sequence of
      `ObservationResponse` messages, each corresponding to a point in time
      within the requested interval.
    """

  @abc.abstractmethod
  def request_action(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.ActionResponse:
    """Applies a control action (e.g., changing setpoints) to the building.

    The implementing class should interpret the `action_request`, apply the
    specified changes to the building model, and return an `ActionResponse`
    indicating the outcome of these actions.

    Args:
      action_request (smart_control_building_pb2.ActionRequest): A protobuf
        message detailing the actions to be taken (e.g., device ID, setpoint
        name, new value).

    Returns:
      smart_control_building_pb2.ActionResponse: A protobuf message confirming
      the status of each requested action (e.g., accepted, rejected).
    """

  @abc.abstractmethod
  def wait_time(self) -> None:
    """Advances the building simulation or waits for real-time progression.

    For simulated buildings, this method typically advances the simulation clock
    by one time step. For physical buildings, it might involve a delay
    corresponding to the environment's step interval.
    """

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the building model to its initial or a new starting state.

    This is crucial for starting new episodes in the RL environment.
    Implementations should restore the building to a consistent baseline state.

    Raises:
      RuntimeError: If resetting the building is not possible or fails.
    """

  @property
  @abc.abstractmethod
  def devices(self) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Provides a list of all controllable and observable devices.

    Each device is described by a `DeviceInfo` protobuf message, which
    includes its ID, observable fields, and actionable setpoints.

    Returns:
      Sequence[smart_control_building_pb2.DeviceInfo]: A list of protobuf
      messages, each describing a device in the building.
    """

  @property
  @abc.abstractmethod
  def zones(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Provides a list of all thermal zones within the building.

    Each zone is described by a `ZoneInfo` protobuf message. This is
    relevant for zone-level control and observation.

    Returns:
      Sequence[smart_control_building_pb2.ZoneInfo]: A list of protobuf
      messages, each describing a thermal zone.
    """

  @property
  @abc.abstractmethod
  def current_timestamp(self) -> pd.Timestamp:
    """Returns the current local time of the building model.

    This timestamp is essential for time-dependent logic within the
    environment and agent, such as scheduling or time-of-day features.

    Returns:
      pd.Timestamp: The current timestamp in the building's local time zone.
    """

  @abc.abstractmethod
  def render(self, path: str) -> None:
    """Renders the current state of the building, e.g., to a file or display.

    The specific nature of rendering (e.g., graphical output, data dump)
    depends on the implementing class.

    Args:
      path (str): A path or identifier specifying where or how to render the
        output.
    """

  @abc.abstractmethod
  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Determines if the building is currently in a "comfort" mode.

    Comfort mode typically implies that HVAC systems are actively maintaining
    conditions suitable for occupancy, potentially based on a schedule.

    Args:
      current_time (pd.Timestamp): The timestamp at which to check comfort mode.

    Returns:
      bool: True if the building is in comfort mode at `current_time`,
      False otherwise.
    """

  @property
  @abc.abstractmethod
  def num_occupants(self) -> int:
    """Returns the current number of occupants in the building.

    Occupancy information can be crucial for demand-driven control strategies.

    Returns:
      int: The total number of occupants currently in the building.
    """

  @property
  @abc.abstractmethod
  def time_step_sec(self) -> float:
    """Returns the duration of a single environment time step in seconds.

    This defines the granularity of control and simulation progression.

    Returns:
      float: The length of one time step in seconds.
    """
