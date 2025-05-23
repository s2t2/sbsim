"""Populates a Reverb replay buffer with initial experiences.

This script is used to generate a "starter" replay buffer by running a
predefined policy (e.g., a baseline schedule-based policy) in the smart
building environment. The collected trajectories are stored in a Reverb
replay buffer, which can then be used to bootstrap the training of a
reinforcement learning agent.

The script handles:
- Environment setup using Gin configuration.
- Creation of a Reverb replay buffer and server.
- Instantiation of a data collection policy (e.g., `SchedulePolicy`).
- Running an actor to collect experiences and add them to the buffer.
- Periodic checkpointing of the replay buffer.
"""

import argparse
import logging
import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.train import actor
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import trajectory

from smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_control.reinforcement_learning.policies.schedule_policy import create_baseline_schedule_policy
from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.config import OUTPUT_DATA_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]",
)
logger = logging.getLogger(__name__)


def populate_replay_buffer(
    buffer_name: str,
    buffer_capacity: int,
    steps_per_run: int,
    num_runs: int,
    sequence_length: int,
    env_gin_config_file_path: str,
) -> reverb_replay_buffer.ReverbReplayBuffer:
  """Populates a Reverb replay buffer using a baseline policy.

  Args:
    buffer_name (str): A unique name for the replay buffer. The buffer data
      will be saved in a subdirectory under
      `smart_control/reinforcement_learning/data/starter_buffers/` using this
      name, along with sequence length and total experience steps.
    buffer_capacity (int): The maximum number of items (trajectories or
      sequences) the replay buffer can hold.
    steps_per_run (int): The number of environment steps the actor will take in
      each data collection run before potential checkpointing or logging.
    num_runs (int): The total number of data collection runs to perform. The
      total number of steps collected will be `num_runs * steps_per_run`.
    sequence_length (int): The length of trajectory sequences to store in the
      replay buffer. This is important for sequence-based learning.
    env_gin_config_file_path (str): Path to the Gin configuration file that
      defines the environment setup.

  Returns:
    reverb_replay_buffer.ReverbReplayBuffer: The populated Reverb replay
    buffer instance.

  Raises:
    FileExistsError: If the directory for the specified `buffer_name` already
      exists, to prevent accidental overwriting of existing buffer data.
  """
  # Construct the full path for saving the replay buffer checkpoints
  total_experience_steps = num_runs * steps_per_run
  buffer_dir_name = (
      f"{buffer_name}_seqlen{sequence_length}_exp{total_experience_steps}"
  )
  buffer_checkpoint_path = os.path.join(OUTPUT_DATA_PATH, buffer_dir_name)
  logger.info("Replay buffer checkpoint path: %s", buffer_checkpoint_path)

  # Ensure the target directory does not already exist to avoid overwriting
  try:
    # os.makedirs requires the parent of the leaf dir for exist_ok=False
    # to work as intended for checking the leaf directory itself.
    # So, we try to create the leaf directory directly.
    os.makedirs(buffer_checkpoint_path)
    logger.info("Created directory for replay buffer: %s", buffer_checkpoint_path)
  except FileExistsError as e:
    logger.error(
        "Buffer path '%s' already exists. This script would overwrite an "
        "existing buffer. Please choose a different 'buffer_name' or remove "
        "the existing directory.",
        buffer_checkpoint_path,
    )
    raise FileExistsError(
        f"Buffer directory {buffer_checkpoint_path} already exists."
    ) from e

  # Set up the environment using Gin configuration
  logger.info(
      "Loading environment from Gin config: %s", env_gin_config_file_path
  )
  collect_py_env = create_and_setup_environment(
      gin_config_file=env_gin_config_file_path,
      metrics_path=None # No metrics needed for buffer population
  )
  collect_tf_env = tf_py_environment.TFPyEnvironment(collect_py_env)

  # Define the data collection policy (e.g., a baseline schedule policy)
  train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)
  _, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_tf_env)
  collection_policy = create_baseline_schedule_policy(collect_tf_env)
  logger.info("Using baseline schedule policy for data collection.")

  # Define the trajectory specification for the replay buffer
  # This must match what the collection_policy and environment produce.
  collect_data_spec = trajectory.Trajectory(
      step_type=time_step_spec.step_type,
      observation=time_step_spec.observation,
      action=action_spec,
      policy_info=collection_policy.info_spec, # Get policy_info from the actual policy
      next_step_type=time_step_spec.step_type,
      reward=time_step_spec.reward,
      discount=time_step_spec.discount,
  )

  # Initialize the ReplayBufferManager and create the Reverb replay buffer
  logger.info("Creating Reverb replay buffer at: %s", buffer_checkpoint_path)
  logger.info(
      "Buffer capacity: %d, Sequence length: %d",
      buffer_capacity,
      sequence_length,
  )
  replay_manager = ReplayBufferManager(
      data_spec=collect_data_spec,
      capacity=buffer_capacity,
      checkpoint_dir=buffer_checkpoint_path,
      sequence_length=sequence_length,
  )
  rb_instance, rb_observer = replay_manager.create_replay_buffer()

  # Set up observers for monitoring data collection
  status_observer = PrintStatusObserver(
      status_interval_steps=steps_per_run // 10 or 1, # Log status ~10 times per run
      environment=collect_tf_env,
      replay_buffer_instance=rb_instance,
  )
  # The Reverb observer (rb_observer) adds data to the buffer.
  # CompositeObserver allows using multiple observers.
  combined_observers = CompositeObserver([status_observer, rb_observer])

  # Create the actor for data collection
  logger.info("Setting up data collection actor.")
  # The actor interacts with the Python environment directly.
  collect_actor_instance = actor.Actor(
      env=collect_tf_env.pyenv, # Pass the underlying PyEnvironment
      policy=py_tf_eager_policy.PyTFEagerPolicy(collection_policy, use_tf_function=True),
      train_step=train_step_counter,
      steps_per_run=steps_per_run,
      observers=[combined_observers],
  )

  # Perform data collection runs
  logger.info(
      "Starting data collection: %d runs, %d steps per run (total %d steps).",
      num_runs,
      steps_per_run,
      total_experience_steps,
  )
  for run_num in range(num_runs):
    logger.info(
        "Run %d/%d (Total steps collected so far: %d)",
        run_num + 1,
        num_runs,
        run_num * steps_per_run,
    )
    collect_actor_instance.run() # Collect `steps_per_run` trajectories

    # Checkpoint the replay buffer after each run
    logger.info(
        "Completed run %d/%d. Checkpointing replay buffer...",
        run_num + 1,
        num_runs,
    )
    # The Reverb server handles checkpointing internally if configured.
    # Explicit checkpointing via client can also be done if needed.
    # replay_manager.replay_buffer.py_client.checkpoint() # If direct control desired

  # Final checkpoint and logging of buffer size
  logger.info(
      "Data collection complete. Total steps: %d.", total_experience_steps
  )
  # Reverb server checkpoints automatically at intervals or on shutdown if configured.
  # Explicit final checkpoint:
  if replay_manager.replay_buffer:
    replay_manager.replay_buffer.py_client.checkpoint()
  logger.info(
      "Final replay buffer size: %d frames.", replay_manager.num_frames()
  )

  return rb_instance


if __name__ == "__main__":
  # Default Gin configuration file for the environment
  default_config_path = os.path.join(CONFIG_PATH, "sim_config_1_day.gin")

  parser = argparse.ArgumentParser(
      description="Populate a Reverb replay buffer with initial exploration "
                  "data using a baseline schedule policy."
  )
  parser.add_argument(
      "--buffer_name", # Changed to snake_case for consistency
      type=str,
      required=True,
      help="Name for the replay buffer. Data will be saved in a subdirectory "
           "under 'smart_control/reinforcement_learning/data/starter_buffers/'.",
  )
  parser.add_argument(
      "--capacity",
      type=int,
      default=50000,
      help="Maximum capacity of the replay buffer. Default: 50000.",
  )
  parser.add_argument(
      "--steps_per_run",
      type=int,
      default=288, # e.g., 1 day of 5-minute intervals
      help="Number of environment steps per actor run. Default: 288.",
  )
  parser.add_argument(
      "--num_runs",
      type=int,
      default=7, # e.g., 1 week of data
      help="Number of actor runs to perform. Total steps = num_runs * "
           "steps_per_run. Default: 7.",
  )
  parser.add_argument(
      "--sequence_length",
      type=int,
      default=2,
      help="Length of trajectory sequences to store in the replay buffer. "
           "Default: 2.",
  )
  parser.add_argument(
      "--env_gin_config_file_path",
      type=str,
      default=default_config_path,
      help="Path to the Gin configuration file for the environment. "
           f"Default: {default_config_path}",
  )
  args = parser.parse_args()

  populate_replay_buffer(
      buffer_name=args.buffer_name,
      buffer_capacity=args.capacity,
      steps_per_run=args.steps_per_run,
      num_runs=args.num_runs,
      sequence_length=args.sequence_length,
      env_gin_config_file_path=args.env_gin_config_file_path,
  )
  logger.info("Replay buffer population script finished.")
