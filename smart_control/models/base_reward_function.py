"""Abstract base class for defining reward functions in smart building control.

This module provides `BaseRewardFunction`, an abstract interface for functions
that calculate a scalar reward signal based on the state and performance of a
building. The reward function is a critical component in reinforcement learning,
guiding the agent's learning process by quantifying the desirability of
different outcomes.

Implementing classes should take raw building performance data (e.g., energy
consumption, occupant comfort metrics, carbon emissions) and transform it into
a single numerical reward value that the RL agent aims to maximize.

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

from smart_control.proto import smart_control_reward_pb2


class BaseRewardFunction(metaclass=abc.ABCMeta):
  """Abstract interface for calculating rewards from building performance data.

  This class defines the method that concrete reward functions must implement.
  The primary role of a reward function is to take a `RewardInfo` message,
  which contains various metrics about the building's state and energy usage,
  and compute a scalar reward. This reward signals to the RL agent how well
  it is performing.

  Conceptual Example:
    A reward function that balances energy cost and comfort:

    ```python
    class EnergyAndComfortReward(BaseRewardFunction):
        def __init__(self, energy_cost_weight: float, comfort_penalty_weight: float):
            self._cost_weight = energy_cost_weight
            self._comfort_weight = comfort_penalty_weight
            # Assume an energy cost model is also provided or accessible
            self._energy_cost_model = ... # e.g., an instance of BaseEnergyCost
            self._occupancy_model = ...   # e.g., an instance of BaseOccupancy

        def compute_reward(self, reward_info: smart_control_reward_pb2.RewardInfo):
            # Calculate energy cost from reward_info (e.g., using energy_rate)
            # This is a simplified example; actual cost calculation would use
            # the energy_cost_model with start/end times from reward_info.
            total_energy_cost = 0.0
            for energy_use in reward_info.energy_consumption.energy_uses:
                # Simplified: actual cost needs start/end times and rates
                # from an energy cost model.
                # total_energy_cost += self._energy_cost_model.cost(...)
                pass # Placeholder for actual energy cost calculation

            # Calculate comfort penalty (e.g., deviation from setpoints)
            # This would involve getting temperature observations and setpoints
            # from reward_info or related observation data.
            comfort_penalty = 0.0
            # ... comfort penalty calculation ...

            # Combine into a scalar reward
            # Agent reward is typically maximized, so costs/penalties are negative
            agent_reward_value = (-self._cost_weight * total_energy_cost -
                                  self._comfort_weight * comfort_penalty)

            return smart_control_reward_pb2.RewardResponse(
                agent_reward_value=agent_reward_value,
                # Populate other fields in RewardResponse as needed
            )
    ```
  """

  @abc.abstractmethod
  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Calculates a scalar reward based on building performance metrics.

    Implementations should process the input `reward_info` (which contains
    data like energy consumption, occupancy, comfort metrics, etc.) and
    return a `RewardResponse` protobuf message. The key field in the
    response is `agent_reward_value`, the scalar reward for the agent.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): A protobuf message
        containing the raw data and metrics from the building environment that
        are relevant for calculating the reward.

    Returns:
      smart_control_reward_pb2.RewardResponse: A protobuf message containing
      the calculated `agent_reward_value` and potentially other detailed
      components of the reward (like energy cost, comfort penalty, etc.).
    """
