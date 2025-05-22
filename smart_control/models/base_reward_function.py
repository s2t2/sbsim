"""Base class for smart buildings reward function.

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

from smart_control.proto import smart_control_reward_pb2


class BaseRewardFunction(metaclass=abc.ABCMeta):
  """Abstract base class for defining the reward function in an RL environment.

  This class provides an interface for calculating a scalar reward signal that
  guides the reinforcement learning agent's behavior. The reward is computed
  based on various aspects of the building's performance and state, which are
  provided via the `RewardInfo` protobuf message.

  Implementations of this class will define the specific logic for how these
  diverse factors (e.g., energy consumption, occupant comfort, operational
  costs, carbon emissions) are weighted and combined into a single reward value.
  The output is a `RewardResponse` protobuf, which includes the final agent
  reward and can also carry disaggregated components of the reward for analysis.
  """

  @abc.abstractmethod
  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Computes the reward based on the provided building performance data.

    Args:
      reward_info: A `smart_control_reward_pb2.RewardInfo` protobuf message
        populated by the `BaseBuilding` implementation. This message contains
        various metrics reflecting the building's state and performance over
        the last control interval (e.g., energy usage by different systems,
        comfort metrics like temperature deviations, occupancy levels).

    Returns:
      A `smart_control_reward_pb2.RewardResponse` protobuf message. This
      includes:
      - `agent_reward_value`: The final scalar reward for the RL agent.
      - Other fields that can store disaggregated components of the reward,
        such as calculated energy costs, comfort penalties, carbon costs, etc.,
        which are useful for logging and detailed performance analysis.
    """
