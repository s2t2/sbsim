"""Utilities for reinforcement learning metrics and trajectory handling.

This module provides functions for constructing TF-Agents trajectories from
environment steps and policy actions, and for evaluating a policy's performance
by computing its average return over multiple episodes in a given environment.
"""

import logging
import time
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd # For pd.Timestamp in example
from tf_agents.environments import py_environment # For type hinting environment
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE

logger = logging.getLogger(__name__)


def get_trajectory(
    time_step: ts.TimeStep,
    current_action_step: policy_step.PolicyStep # Renamed for clarity
) -> trajectory.Trajectory:
  """Constructs a TF-Agents `Trajectory` object from a time step and action.

  This function takes the current `time_step` from the environment (which includes
  the observation, reward, discount, and step type) and the `current_action_step`
  (which includes the action taken by the policy and policy-specific info)
  to form a complete `Trajectory` object. It correctly identifies whether the
  trajectory represents the first step, a middle step, or the last step of an
  episode.

  Args:
    time_step: A `tf_agents.trajectories.time_step.TimeStep` namedtuple
      containing the observation, reward, discount, and step type from the
      environment at the current point in time.
    current_action_step: A `tf_agents.trajectories.policy_step.PolicyStep`
      namedtuple containing the action taken by the policy and any associated
      policy information.

  Returns:
    A `tf_agents.trajectories.trajectory.Trajectory` object representing the
    transition.
  """
  observation = time_step.observation
  action = current_action_step.action
  # Assuming policy_info is empty for this context or taken from action_step if available
  policy_info = current_action_step.info if hasattr(current_action_step, 'info') else ()
  reward = time_step.reward
  discount = time_step.discount

  if time_step.is_first():
    # For the first step of an episode.
    return trajectory.first(
        observation=observation,
        action=action,
        policy_info=policy_info,
        reward=reward,
        discount=discount
    )
  elif time_step.is_last():
    # For the last step of an episode.
    return trajectory.last(
        observation=observation,
        action=action,
        policy_info=policy_info,
        reward=reward,
        discount=discount
    )
  else:
    # For intermediate steps in an episode.
    return trajectory.mid(
        observation=observation,
        action=action,
        policy_info=policy_info,
        reward=reward,
        discount=discount
    )


def compute_avg_return(
    environment: py_environment.PyEnvironment, # More specific type
    policy: py_policy.PyPolicy,
    num_episodes: int = 1,
    time_zone: str = DEFAULT_TIME_ZONE,
    trajectory_observers: Optional[List[Callable[[trajectory.Trajectory], None]]] = None,
    num_steps: int = 6,
) -> Tuple[float, List[Tuple[pd.Timestamp, float]]]: # Return type hint improved
  """Evaluates a policy by computing its average return over multiple episodes.

  This function runs the given `policy` in the specified `environment` for
  `num_episodes`, each for a maximum of `num_steps`. It calculates the total
  reward (return) for each episode and then averages these returns.
  During evaluation, it can also invoke optional `trajectory_observers` for
  each step. Detailed logging of rewards and timing per step is performed.

  Args:
    environment: The `tf_agents.environments.PyEnvironment` instance in which
      to evaluate the policy.
    policy: The `tf_agents.policies.PyPolicy` instance to be evaluated.
    num_episodes: The total number of episodes to run for the evaluation.
      Defaults to 1.
    time_zone: The timezone string (e.g., 'US/Pacific') used for logging
      simulation timestamps. Defaults to `DEFAULT_TIME_ZONE`.
    trajectory_observers: An optional list of callable observers. Each observer
      will be called with the trajectory of each step taken during evaluation.
      This can be used for detailed logging or custom metric calculation.
      Defaults to `None`.
    num_steps: The maximum number of steps to run within each evaluation
      episode. Defaults to 6.

  Returns:
    A tuple containing:
      - avg_return (float): The average total reward (return) achieved by the
        policy across all evaluation episodes.
      - return_by_simtime (List[Tuple[pd.Timestamp, float]]): A list where each
        element is a tuple `(simulation_timestamp, cumulative_episode_return)`.
        This tracks the cumulative return at each step's simulation time across
        all episodes.

  Example:
    ```python
    # Assuming `my_eval_env` is a PyEnvironment and `my_eval_policy` is a PyPolicy
    # avg_return, return_details = compute_avg_return(
    #     environment=my_eval_env,
    #     policy=my_eval_policy,
    #     num_episodes=10,
    #     num_steps=288 # e.g., steps in a day for a 5-min step interval
    # )
    # print(f"Average return over 10 episodes: {avg_return}")
    ```
  """
  total_return = 0.0
  return_by_simtime: List[Tuple[pd.Timestamp, float]] = []

  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0
    t0 = time.time()
    epoch = t0
    step_id = 0
    execution_times = []

    for _ in range(num_steps):
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)

      if trajectory_observers is not None:
        traj = get_trajectory(time_step, action_step)
        for observer in trajectory_observers:
          observer(traj)

      episode_return += time_step.reward
      t1 = time.time()
      dt = t1 - t0
      episode_seconds = t1 - epoch
      execution_times.append(dt)
      sim_time = environment.pyenv.envs[
          0
      ].current_simulation_timestamp.tz_convert(time_zone)

      return_by_simtime.append([sim_time, episode_return])

      logger.info(
          "[Step %d] [Sim Time: %s] [Reward: %.2f] [Return: %.2f] "
          "[Mean Step Time: %.2fs] [Episode Time: %.2fs]",
          step_id,
          sim_time.strftime("%Y-%m-%d %H:%M"),
          time_step.reward,
          episode_return,
          np.mean(execution_times),
          episode_seconds,
      )

      t0 = t1
      step_id += 1
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return, return_by_simtime
