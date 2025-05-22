"""Provides an RL observer for printing training status updates to the logger.

This module defines the `PrintStatusObserver` class, which logs information
about the training progress at specified intervals. This includes details like
step counts, simulation time, rewards, execution speed, and replay buffer size.
"""

import logging
import time # Import the time module

import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.replay_buffers import replay_buffer
from tf_agents.trajectories import trajectory as trajectory_lib


from smart_control.reinforcement_learning.observers.base_observer import Observer
from smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE

logger = logging.getLogger(__name__)


class PrintStatusObserver(Observer):
  """An observer that logs training status information periodically.

  This observer monitors the RL training process and prints status updates
  to the configured logger. The logged information includes:
  - Current training step and total steps in the episode.
  - Percentage of episode completion.
  - Current simulation timestamp from the environment.
  - Reward obtained in the last step.
  - Cumulative reward for the current episode.
  - Total execution time since the last reset.
  - Mean execution time per step.
  - Current size of the replay buffer (if provided).
  """

  def __init__(
      self,
      status_interval_steps: int = 1,
      environment: py_environment.PyEnvironment | None = None,
      replay_buffer: replay_buffer.ReplayBuffer | None = None,
      time_zone: str = DEFAULT_TIME_ZONE,
  ):
    """Initializes the PrintStatusObserver.

    Args:
      status_interval_steps: The interval (in training steps) at which status
        updates will be printed. Defaults to 1 (print every step).
      environment: An optional `tf_agents.environments.PyEnvironment` instance.
        Used to access simulation time and episode step counts. If not provided,
        some status information related to environment progress will be unavailable.
      replay_buffer: An optional `tf_agents.replay_buffers.ReplayBuffer` instance.
        If provided, its current size will be logged.
      time_zone: The timezone string (e.g., "America/Los_Angeles", "UTC") used
        to display the simulation timestamp. Defaults to `DEFAULT_TIME_ZONE`.
    """
    self._counter = 0
    self._status_interval_steps = status_interval_steps
    self._environment = environment
    self._cumulative_reward = 0.0
    self._replay_buffer = replay_buffer
    self._time_zone = time_zone

    self._start_time: float | None = None # Store time as float (timestamp)
    self._num_timesteps_in_episode = 0
    if self._environment and hasattr(self._environment, 'pyenv') and \
       self._environment.pyenv.envs and \
       hasattr(self._environment.pyenv.envs[0], '_num_timesteps_in_episode'):
      self._num_timesteps_in_episode = self._environment.pyenv.envs[0]._num_timesteps_in_episode # pylint: disable=protected-access

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Processes the trajectory and logs status if the interval is met.

    This method accumulates rewards, increments a step counter, and if the
    current step is a multiple of `status_interval_steps`, it calculates
    and logs various performance and progress metrics.

    Args:
      trajectory: The `tf_agents.trajectories.trajectory.Trajectory` object
        from the current training step.
    """
    # Ensure trajectory.reward is a scalar float or can be converted.
    # For batched trajectories, this might need aggregation (e.g., mean).
    # Assuming trajectory.reward is scalar or a 0-d tensor here.
    try:
      reward_value = float(trajectory.reward)
    except TypeError:
      # Handle cases where reward might be a more complex structure (e.g. array in batch)
      # For simplicity, trying to get a single value if possible, or logging a warning.
      if hasattr(trajectory.reward, 'numpy') and trajectory.reward.numpy().size == 1:
        reward_value = float(trajectory.reward.numpy().item())
      else:
        logger.warning("Cannot extract scalar reward from trajectory for PrintStatusObserver.")
        reward_value = 0.0 # Default or skip if reward structure is unexpected

    self._cumulative_reward += reward_value
    self._counter += 1
    if self._start_time is None:
      self._start_time = time.time() # Use time.time() for float timestamp

    if self._counter % self._status_interval_steps == 0 and self._environment and \
       hasattr(self._environment, 'pyenv') and self._environment.pyenv.envs and \
       hasattr(self._environment.pyenv.envs[0], 'current_simulation_timestamp') and \
       hasattr(self._environment.pyenv.envs[0], '_step_count'):

      current_time_float = time.time()
      execution_duration_seconds = current_time_float - self._start_time
      mean_execution_time_seconds = execution_duration_seconds / self._counter

      # Safely access environment attributes
      # pylint: disable=protected-access
      sim_time = self._environment.pyenv.envs[0].current_simulation_timestamp.tz_convert(self._time_zone)
      current_env_step = self._environment.pyenv.envs[0]._step_count
      # pylint: enable=protected-access

      percent_complete = 0
      if self._num_timesteps_in_episode > 0 :
        # Use current_env_step for percent_complete if it reflects episode progress
        percent_complete = int(100.0 * (current_env_step / self._num_timesteps_in_episode))


      rb_string = "Replay Buffer Size: N/A"
      if self._replay_buffer is not None and hasattr(self._replay_buffer, 'num_frames'):
        rb_size = self._replay_buffer.num_frames()
        rb_string = f"Replay Buffer Size: {rb_size}"

      logger.info(
          "[Step %d of %d (%d%%)] [Sim Time: %s] [Reward: %.2f] "
          "[Cum Reward: %.2f]",
          current_env_step,
          self._num_timesteps_in_episode,
          percent_complete,
          sim_time.strftime("%Y-%m-%d %H:%M"),
          reward_value,
          self._cumulative_reward,
      )

      # Format execution_duration_seconds into a more readable string if desired
      # For example, convert to HH:MM:SS or similar
      formatted_execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_duration_seconds))

      logger.info(
          "[Exec Time: %s] [Mean Exec Time: %.2fs/step] [%s]",
          formatted_execution_time,
          mean_execution_time_seconds,
          rb_string,
      )

  def reset(self) -> None:
    """Resets the observer's internal state for a new episode.

    This clears the step counter, cumulative reward, and resets the
    start time. It is typically called at the beginning of each new
    training or evaluation episode.
    """
    self._counter = 0
    self._cumulative_reward = 0.0
    self._start_time = None
