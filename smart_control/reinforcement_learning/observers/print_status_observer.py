"""Observer for printing agent and environment status during training/evaluation.

This module defines the `PrintStatusObserver`, an implementation of the
`Observer` interface that logs key metrics and status information to the console
(or a configured logger). It helps in monitoring the progress of an agent's
interaction with the environment.
"""

import logging
from typing import Optional

import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.replay_buffers import replay_buffer
from tf_agents.trajectories import trajectory as trajectory_lib

from smart_control.reinforcement_learning.observers.base_observer import Observer
from smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE

logger = logging.getLogger(__name__)


class PrintStatusObserver(Observer):
  """Logs status information about RL agent interaction with an environment.

  This observer tracks and periodically prints:
  - Current simulation step and episode progress.
  - Current simulation time.
  - Instantaneous and cumulative rewards.
  - Execution time for steps.
  - Replay buffer size (if provided).

  Attributes:
    _counter (int): Number of trajectories processed since the last reset.
    _status_interval_steps (int): Frequency (in steps) at which to print status.
    _environment (Optional[py_environment.PyEnvironment]): The environment
      being observed. Used to fetch simulation time and episode details.
    _cumulative_reward (float): Reward accumulated since the last reset.
    _replay_buffer (Optional[replay_buffer.ReplayBuffer]): The replay buffer
      used by the agent, to report its size.
    _time_zone (str): Time zone for displaying simulation timestamps.
    _start_time (Optional[pd.Timestamp]): Wall clock time when processing started
      after the last reset.
    _num_timesteps_in_episode (int): Total number of timesteps expected in an
      episode.
  """

  def __init__(
      self,
      status_interval_steps: int = 100,
      environment: Optional[py_environment.PyEnvironment] = None,
      replay_buffer_instance: Optional[replay_buffer.ReplayBuffer] = None,
      time_zone: str = DEFAULT_TIME_ZONE,
  ):
    """Initializes the PrintStatusObserver.

    Args:
      status_interval_steps (int): The interval (number of agent steps) at
        which to log status updates.
      environment (Optional[py_environment.PyEnvironment]): The TF-Agents
        Python environment that the agent is interacting with. This is used to
        access simulation-specific information like current time and total
        steps in an episode.
      replay_buffer_instance (Optional[replay_buffer.ReplayBuffer]): The
        replay buffer being used for training. If provided, its size will be
        logged.
      time_zone (str): The time zone to use when displaying timestamps.
    """
    self._counter: int = 0
    self._status_interval_steps: int = status_interval_steps
    self._environment: Optional[py_environment.PyEnvironment] = environment
    self._cumulative_reward: float = 0.0
    self._replay_buffer: Optional[replay_buffer.ReplayBuffer] = (
        replay_buffer_instance
    )
    self._time_zone: str = time_zone

    self._start_time: Optional[pd.Timestamp] = None
    self._num_timesteps_in_episode: int = 0
    if self._environment and hasattr(self._environment, "pyenv"):
      # Accessing underlying PyEnvironment, common in TF-Agents wrappers.
      # Assumes the first env in a batched env has relevant episode info.
      # This might need adjustment if the environment structure is different.
      try:
        # Attempt to get num_timesteps_in_episode from the custom attribute
        # of the underlying custom environment.
        self._num_timesteps_in_episode = self._environment.pyenv.envs[
            0
        ]._num_timesteps_in_episode
      except (AttributeError, IndexError) as e:
        logger.warning(
            "Could not determine _num_timesteps_in_episode from environment: %s. "
            "Percentage complete will not be accurate.",
            e,
        )
        self._num_timesteps_in_episode = -1 # Indicates unknown
    else:
        logger.warning(
            "Environment not provided or not structured as expected. "
            "Some status information may be unavailable."
        )
        self._num_timesteps_in_episode = -1

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Processes a trajectory and logs status if the interval is met.

    Args:
      trajectory (trajectory_lib.Trajectory): The trajectory from the agent's
        step in the environment.
    """
    # Ensure reward is scalar if it's a TF tensor or NumPy array.
    reward_value = trajectory.reward
    if hasattr(reward_value, "numpy"): # TF tensor
      reward_value = reward_value.numpy()
    if isinstance(reward_value, (list, tuple, pd.Series)) and len(reward_value) == 1:
        reward_value = reward_value[0]
    elif isinstance(reward_value, pd.DataFrame) and reward_value.size == 1:
        reward_value = reward_value.iloc[0,0]

    self._cumulative_reward += float(reward_value)
    self._counter += 1

    if self._start_time is None:
      self._start_time = pd.Timestamp.now()

    if self._counter % self._status_interval_steps == 0 and self._environment and hasattr(self._environment, "pyenv"):
      execution_time = pd.Timestamp.now() - self._start_time
      mean_execution_time_sec = execution_time.total_seconds() / self._counter

      # Accessing potentially custom attributes of the underlying environment.
      # This part is specific to how the custom environment is structured.
      current_sim_env = self._environment.pyenv.envs[0]
      sim_time_utc = current_sim_env.current_simulation_timestamp
      sim_time_local = sim_time_utc.tz_convert(self._time_zone)

      percent_complete_str = ""
      if self._num_timesteps_in_episode > 0:
        percent_complete = int(
            100.0 * (current_sim_env._step_count / self._num_timesteps_in_episode)
        )
        percent_complete_str = f"{percent_complete}%"


      rb_string = ""
      if self._replay_buffer is not None:
        try:
          rb_size = self._replay_buffer.num_frames()
          rb_string = f"Replay Buffer Size: {rb_size.numpy() if hasattr(rb_size, 'numpy') else rb_size}"
        except Exception as e: # pylint: disable=broad-except
          rb_string = f"Replay Buffer Size: <Error: {e}>"


      logger.info(
          "[Step %d of %s %s] [Sim Time: %s] [Reward: %.2f] "
          "[Cum Reward: %.2f]",
          current_sim_env._step_count,
          str(self._num_timesteps_in_episode) if self._num_timesteps_in_episode > 0 else "N/A",
          percent_complete_str,
          sim_time_local.strftime("%Y-%m-%d %H:%M"),
          float(reward_value),
          self._cumulative_reward,
      )

      logger.info(
          "[Exec Time: %s] [Mean Step Exec Time: %.2fs] [%s]",
          str(execution_time).split('.')[0], # More concise timedelta format
          mean_execution_time_sec,
          rb_string,
      )

  def reset(self) -> None:
    """Resets the observer's internal counters and timers.
    This is typically called at the start of a new episode.
    """
    self._counter = 0
    self._cumulative_reward = 0.0
    self._start_time = None
