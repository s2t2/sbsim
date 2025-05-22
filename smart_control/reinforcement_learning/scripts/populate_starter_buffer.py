"""Populates a Reverb replay buffer with initial experience data.

This script is used to generate a "starter" replay buffer by running a
predefined policy (e.g., a schedule-based baseline policy) in a smart building
control environment. The collected trajectories are stored in a Reverb replay
buffer, which can then be used to bootstrap or accelerate the training of a
reinforcement learning agent.

The script is configurable via command-line arguments for buffer naming,
capacity, collection duration, and environment settings.

Example usage from the command line:
  ```bash
  python -m smart_control.reinforcement_learning.scripts.populate_starter_buffer \
    --buffer-name="my_starter_buffer" \
    --capacity=100000 \
    --steps-per-run=288 \
    --num-runs=10 \
    --sequence-length=2 \
    --env-gin-config-file-path="path/to/your/sim_config.gin"
  ```
This would collect 10 runs of 288 steps each (total 2880 steps/trajectories)
and store them in a buffer named "my_starter_buffer_seqlen2_exp2880".
"""

import argparse
import logging
import os

import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import py_tf_eager_policy # For wrapping TF policies
from tf_agents.train import actor
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import trajectory
from tf_agents.replay_buffers import reverb_replay_buffer # For type hint

from smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_control.reinforcement_learning.policies.schedule_policy import create_baseline_schedule_policy
from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.config import OUTPUT_DATA_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]',
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
  """Creates and populates a Reverb replay buffer with initial exploration data.

  This function initializes a smart building control environment using a Gin
  configuration file, sets up a Reverb replay buffer, and collects experience
  by running a baseline schedule-based policy in the environment for a specified
  number of steps and runs. The collected data is stored in the replay buffer,
  and the buffer is checkpointed periodically and at the end of collection.

  Args:
    buffer_name: A unique name for the replay buffer. This name is used to
      create a subdirectory under `OUTPUT_DATA_PATH` (from config) where the
      buffer's checkpoint files will be stored. The final path also includes
      sequence length and total experience steps.
    buffer_capacity: The maximum number of elements (trajectories or sequences)
      that the replay buffer can hold.
    steps_per_run: The number of environment steps the data collection actor
      will execute in each run (iteration).
    num_runs: The total number of data collection runs to perform. The total
      number of steps collected will be `num_runs * steps_per_run`.
    sequence_length: The length of trajectory sequences to store in the replay
      buffer. This is important for agents that learn from sequences.
    env_gin_config_file_path: The file path to the Gin configuration file that
      defines the smart building environment setup.

  Returns:
    The populated `reverb_replay_buffer.ReverbReplayBuffer` instance.
    The primary side effect is the creation and checkpointing of this buffer
    on disk at the constructed `buffer_path`.

  Raises:
    FileExistsError: If the target directory for the replay buffer (derived
      from `buffer_name`) already exists, to prevent accidental overwriting of
      existing buffer data.
  """
  # Construct the full path for storing the replay buffer checkpoints
  # The path includes the buffer name, sequence length, and total experience.
  total_experience_steps = num_runs * steps_per_run
  buffer_dir_name = f'{buffer_name}_seqlen{sequence_length}_exp{total_experience_steps}'
  buffer_path = os.path.join(OUTPUT_DATA_PATH, buffer_dir_name)
  logger.info('Target replay buffer path: %s', buffer_path)

  # Create directory for the buffer; raise error if it already exists.
  # os.makedirs will create parent dirs if they don't exist.
  # We use a temporary file within the dir to check existence of the final dir itself.
  try:
    # Check if the specific buffer directory already exists
    if os.path.exists(buffer_path):
        raise FileExistsError(
            f"Buffer directory {buffer_path} already exists. Please use another "
            "name or delete the existing directory."
        )
    os.makedirs(buffer_path)
    logger.info('Created directory for replay buffer: %s', buffer_path)
  except FileExistsError as err:
    logger.exception(
        'This buffer path (%s) already exists. This would override the existing '
        'buffer. Please use another name or remove the existing directory.',
        buffer_path
    )
    raise # Re-raise the error to stop execution

  # Load and set up the environment using the provided Gin config file
  logger.info('Loading environment from Gin config: %s', env_gin_config_file_path)
  # `metrics_path=None` because we are only collecting for buffer, not full eval
  py_collect_env = create_and_setup_environment(
      env_gin_config_file_path, metrics_path=None
  )

  # Wrap the Python environment in a TFPyEnvironment for TF-Agents compatibility
  collect_tf_env = tf_py_environment.TFPyEnvironment(py_collect_env)

  # Create a policy for data collection (e.g., a baseline schedule policy)
  # This policy will dictate the actions taken during data gathering.
  train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name='train_step_counter')
  logger.info('Creating baseline schedule policy for data collection.')
  collection_policy = create_baseline_schedule_policy(collect_tf_env)

  # Define the data specification for trajectories stored in the replay buffer.
  # This spec must match the data produced by the collection_policy and collect_tf_env.
  logger.info('Defining trajectory data spec for replay buffer.')
  collect_data_spec = trajectory.from_spec(collection_policy.collect_data_spec)


  # Initialize the ReplayBufferManager and create the Reverb replay buffer
  logger.info('Initializing ReplayBufferManager and creating Reverb replay buffer.')
  logger.info(
      'Buffer capacity: %d, Sequence length: %d, Checkpoint path: %s',
      buffer_capacity, sequence_length, buffer_path
  )
  replay_manager = ReplayBufferManager(
      data_spec=collect_data_spec,
      capacity=buffer_capacity,
      checkpoint_dir=buffer_path, # Pass the specific directory for this buffer
      sequence_length=sequence_length,
  )
  # This creates the Reverb server, table, and the ReplayBuffer object itself.
  replay_buffer, replay_observer = replay_manager.create_replay_buffer()

  # Set up observers for monitoring the collection process
  # PrintStatusObserver logs progress to the console.
  # replay_observer (from ReplayBufferManager) adds trajectories to the Reverb buffer.
  logger.info('Setting up observers for data collection.')
  print_observer = PrintStatusObserver(
      status_interval_steps=10, # Log status every 10 steps
      environment=collect_tf_env, # Pass the TF environment
      replay_buffer=replay_buffer, # For logging buffer size
  )
  # CompositeObserver allows using multiple observers simultaneously.
  observers = CompositeObserver([print_observer, replay_observer])

  # Create the TF-Agents Actor for data collection.
  # The actor uses the Python environment and a PyTFEagerPolicy wrapper for the collection policy.
  logger.info('Setting up data collection actor.')
  # Ensure the policy is wrapped correctly for the Actor if it's a TFPolicy.
  # py_tf_eager_policy.PyTFEagerPolicy is used if collection_policy is a TFPolicy.
  # If it's already a PyPolicy, it can be used directly.
  collect_py_policy = py_tf_eager_policy.PyTFEagerPolicy(
      collection_policy, use_tf_function=True
  )
  collect_actor = actor.Actor(
      env=py_collect_env, # Actor interacts with the Python environment
      policy=collect_py_policy,
      train_step=train_step_counter, # A TF variable to count training steps (used by Actor)
      steps_per_run=steps_per_run, # Number of steps per actor.run() call
      observers=[observers] # Pass the composite observer
  )

  # Start the data collection process
  logger.info(
      'Starting collection for %d runs of %d steps each',
      num_runs,
      steps_per_run,
  )
  total_steps = 0

  for current_run in range(num_runs):
    # Run collection
    logger.info(
        'Run %d/%d (total steps so far: %d)',
        current_run + 1,
        num_runs,
        total_steps,
    )
    collect_actor.run()

    # Update total steps
    total_steps += steps_per_run

    # Checkpoint buffer periodically
    logger.info(
        'Completed run %d/%d. Checkpointing buffer...',
        current_run + 1,
        num_runs,
    )
    replay_buffer.py_client.checkpoint()

  # Final checkpoint and stats
  logger.info(
      'Completed all runs, total steps: %d. '
      'Checkpointing buffer one last time...',
      total_steps,
  )

  replay_buffer.py_client.checkpoint()
  logger.info('Final replay buffer size: %d frames', replay_buffer.num_frames())

  return replay_buffer


if __name__ == '__main__':

  config_filepath = os.path.join(CONFIG_PATH, 'sim_config_1_day.gin')

  # fmt: off
  # pylint: disable=line-too-long
  parser = argparse.ArgumentParser(description='Populate a replay buffer with initial exploration data')
  parser.add_argument('--buffer-name', type=str, required=True, help='Name to identify the saved replay buffer')
  parser.add_argument('--capacity', type=int, default=50000, help='Replay buffer capacity')
  parser.add_argument('--steps-per-run', type=int, default=100, help='Number of steps per actor run')
  parser.add_argument('--num-runs', type=int, default=5, help='Number of actor runs to perform')
  parser.add_argument('--sequence-length', type=int, default=2, help='Sequence length for the replay buffer')
  parser.add_argument('--env-gin-config-file-path', type=str, default=config_filepath, help='Environment config file')
  # pylint: enable=line-too-long
  # fmt: on
  args = parser.parse_args()

  populate_replay_buffer(
      buffer_name=args.buffer_name,
      buffer_capacity=args.capacity,
      steps_per_run=args.steps_per_run,
      num_runs=args.num_runs,
      sequence_length=args.sequence_length,
      env_gin_config_file_path=args.env_gin_config_file_path,
  )
