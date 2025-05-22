"""Base class that extends functionality outside of the building.

The base class should be extended by the simulation and actual buildings.

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
from typing import Sequence

import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2


class BaseBuilding(metaclass=abc.ABCMeta):
  """Abstract base class defining the interface for a controllable building.

  This class provides a standardized way for reinforcement learning (RL)
  environments to interact with different building simulations or real-world
  buildings. Implementations of this interface should handle the specifics of
  communication, state representation, and action execution for the target
  building system.

  Key responsibilities of implementing classes:
  - Provide access to current building state (observations).
  - Allow actions (setpoint changes) to be sent to the building.
  - Manage simulation time and episode resets.
  - Expose building metadata like device and zone information.
  - Supply data necessary for reward calculation.
  """

  @property
  @abc.abstractmethod
  def reward_info(self) -> smart_control_reward_pb2.RewardInfo:
    """Provides data needed to compute the instantaneous reward.

    This property should return a `RewardInfo` protobuf message containing
    all relevant metrics from the building's current state that are required
    by the `BaseRewardFunction` to calculate the agent's reward for the
    current timestep (e.g., energy consumption, comfort violations).

    Returns:
      A `smart_control_reward_pb2.RewardInfo` protobuf message.
    """

  @abc.abstractmethod
  def request_observations(
      self, observation_request: smart_control_building_pb2.ObservationRequest
  ) -> smart_control_building_pb2.ObservationResponse:
    """Queries the building for its current state (observations).

    Args:
      observation_request: A `ObservationRequest` protobuf specifying which
        device fields (measurements) to retrieve. The timestamp in this request
        indicates the time for which observations are requested.

    Returns:
      A `ObservationResponse` protobuf containing the requested observations
      and their validity.
    """

  @abc.abstractmethod
  def request_observations_within_time_interval(
      self,
      observation_request: smart_control_building_pb2.ObservationRequest,
      start_timestamp: pd.Timestamp,
      end_timestamp: pd.Timestamp,
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Queries for a sequence of observations within a specified time interval.

    This method is typically used to fetch historical data from the building.
    The `observation_request` specifies *which* data points to retrieve,
    while `start_timestamp` and `end_timestamp` define the period.

    Args:
      observation_request: A `ObservationRequest` protobuf specifying the
        device fields to retrieve. The timestamp in this request is usually
        ignored in favor of the interval.
      start_timestamp: The beginning of the time interval (inclusive).
      end_timestamp: The end of the time interval (inclusive or exclusive,
        depending on implementation).

    Returns:
      A sequence of `ObservationResponse` protobufs, each corresponding to a
      timestep within the requested interval, ordered chronologically.
    """

  @abc.abstractmethod
  def request_action(
      self, action_request: smart_control_building_pb2.ActionRequest
  ) -> smart_control_building_pb2.ActionResponse:
    """Sends an action command to the building to change setpoints.

    Args:
      action_request: An `ActionRequest` protobuf containing the desired
        setpoint changes for various devices. The timestamp in the request
        indicates when the action should ideally be applied.

    Returns:
      An `ActionResponse` protobuf indicating the outcome of the action
      (e.g., whether setpoints were accepted, rejected, or resulted in errors).
    """

  @abc.abstractmethod
  def wait_time(self) -> None:
    """Advances the building simulation or waits for real-world time progression.

    This method is called after each action to move to the next timestep.
    In a simulation, this typically involves advancing the simulation clock by
    `time_step_sec`. In a real-world deployment, it might involve a literal
    wait or check if sufficient time has passed.
    """

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the building to a known initial state for a new episode.

    This is crucial for episodic RL tasks. Implementations should ensure the
    building (simulated or real, if applicable) is returned to a consistent
    starting point.

    Raises:
      RuntimeError: If resetting the building to a valid initial state is
        impossible or fails.
    """

  @property
  @abc.abstractmethod
  def devices(self) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Provides a list of all devices that can be queried and/or controlled.

    Returns:
      A sequence of `DeviceInfo` protobuf messages, each describing a device,
      its ID, observable fields (sensors/measurements), and actionable fields
      (setpoints).
    """

  @property
  @abc.abstractmethod
  def zones(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Provides a list of thermal zones in the building relevant to control.

    Returns:
      A sequence of `ZoneInfo` protobuf messages, each describing a zone,
      its ID, and potentially associated devices or properties.
    """

  @property
  @abc.abstractmethod
  def current_timestamp(self) -> pd.Timestamp:
    """Returns the current simulation or real-world timestamp of the building.

    This timestamp should reflect the building's local time.

    Returns:
      A `pandas.Timestamp` object representing the current time.
    """

  @abc.abstractmethod
  def render(self, path: str) -> None:
    """Renders the current state of the building, typically for visualization.

    Implementations might generate diagrams, charts, or save state data
    to the specified file path.

    Args:
      path: A string representing the file system path where rendering
        output should be saved.
    """

  @abc.abstractmethod
  def is_comfort_mode(self, current_time: pd.Timestamp) -> bool:
    """Determines if the building is currently in a comfort-providing mode.

    "Comfort mode" typically corresponds to scheduled occupied hours where HVAC
    systems aim to maintain comfortable conditions for occupants.

    Args:
      current_time: The `pandas.Timestamp` for which to check the comfort mode.

    Returns:
      True if the building is considered to be in comfort mode at the given
      time, False otherwise.
    """

  @property
  @abc.abstractmethod
  def num_occupants(self) -> int:
    """Returns the current number of occupants in the building.

    This could be an actual count from sensors or an estimate based on schedules
    or models.

    Returns:
      The integer number of occupants.
    """

  @property
  @abc.abstractmethod
  def time_step_sec(self) -> float:
    """Returns the duration of a single environment step in seconds.

    This value defines the interval at which the RL agent interacts with the
    environment (i.e., takes an action and receives an observation).

    Returns:
      The step duration in seconds as a float.
    """
