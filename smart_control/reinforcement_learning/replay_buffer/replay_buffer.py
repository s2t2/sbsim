"""Manages Reverb replay buffers for reinforcement learning agents in TF-Agents.

This module provides the `ReplayBufferManager` class, a utility designed to
streamline the creation, configuration, management, and usage of Reverb-backed
replay buffers. Reverb is a high-performance system for experience replay,
crucial for off-policy reinforcement learning algorithms.
"""

import logging
from typing import Optional, Tuple

import reverb
import tensorflow as tf
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec # For type hinting data_spec

logger = logging.getLogger(__name__)


class ReplayBufferManager:
  """Manages the lifecycle and interaction with a Reverb replay buffer.

  This class encapsulates the setup of a Reverb server, table, checkpointer,
  and the TF-Agents `ReverbReplayBuffer`. It provides methods to:
  - Create a new replay buffer or load an existing one from a checkpoint.
  - Obtain an observer for adding trajectory data to the buffer.
  - Generate a `tf.data.Dataset` for sampling experiences for agent training.
  - Query buffer status (e.g., number of frames).
  - Clear and close the replay buffer and server.

  Example:
    ```python
    from tf_agents.specs import array_spec
    from tf_agents.trajectories import trajectory

    # Define the data specification for trajectories
    data_spec = trajectory.Trajectory(
        step_type=array_spec.ArraySpec(shape=(), dtype=int, name='step_type'),
        observation=array_spec.ArraySpec(shape=(4,), dtype=float, name='obs'),
        action=array_spec.ArraySpec(shape=(1,), dtype=float, name='act'),
        policy_info=(),
        next_step_type=array_spec.ArraySpec(shape=(), dtype=int, name='next_step_type'),
        reward=array_spec.ArraySpec(shape=(), dtype=float, name='reward'),
        discount=array_spec.ArraySpec(shape=(), dtype=float, name='discount')
    )

    rb_manager = ReplayBufferManager(
        data_spec=data_spec,
        capacity=100000,
        checkpoint_dir="/tmp/my_replay_buffer_checkpoints",
        sequence_length=2
    )

    # Create or load the buffer and observer
    # replay_buffer, observer = rb_manager.create_replay_buffer()
    # Alternatively, to load if checkpoint exists, or create if not:
    # try:
    #     replay_buffer, observer = rb_manager.load_replay_buffer()
    # except tf.errors.NotFoundError: # Or other relevant Reverb/TF errors
    #     print("Checkpoint not found, creating new replay buffer.")
    #     replay_buffer, observer = rb_manager.create_replay_buffer()

    replay_buffer, observer = rb_manager.get_replay_buffer_and_observer()


    # Get a dataset for training
    dataset = rb_manager.get_dataset(batch_size=64)

    # ... use observer to add data, dataset for training ...

    # Close the replay buffer when done
    rb_manager.close()
    ```
  """

  def __init__(
      self,
      data_spec: tensor_spec.TensorSpec,
      capacity: int,
      checkpoint_dir: str,
      sequence_length: int = 2
  ):
    """Initializes the ReplayBufferManager.

    Args:
      data_spec: A `tf_agents.specs.TensorSpec` or a nest of specs, defining
        the structure of trajectories to be stored in the replay buffer (e.g.,
        as obtained from `agent.collect_data_spec`).
      capacity: The maximum number of elements (trajectories or sequences)
        the replay buffer can hold.
      checkpoint_dir: A string path to the directory where replay buffer
        checkpoints will be saved and from which they can be loaded.
      sequence_length: The length of sequences to store and sample from the
        buffer. This is important for agents that learn from sequences of
        experiences (e.g., recurrent policies). Defaults to 2.
    """
    self.data_spec = data_spec
    self.capacity = capacity
    self.checkpoint_dir = checkpoint_dir
    self.sequence_length = sequence_length
    self.table_name = "uniform_table" # Default table name for Reverb
    self._is_initialized = False
    self.server: Optional[reverb.Server] = None
    self.replay_buffer: Optional[reverb_replay_buffer.ReverbReplayBuffer] = None
    self.observer: Optional[reverb_utils.ReverbAddTrajectoryObserver] = None

  def create_replay_buffer(self) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver
  ]:
    """Creates a new Reverb replay buffer, server, and trajectory observer.

    This method configures and starts a local Reverb server, defines a table
    for storing trajectories, sets up a checkpointer for persistence,
    initializes the `ReverbReplayBuffer` to interact with this server, and
    creates a `ReverbAddTrajectoryObserver` for conveniently adding data
    to the buffer.

    Returns:
      A tuple containing:
        - replay_buffer: The initialized `ReverbReplayBuffer` instance.
        - observer: The `ReverbAddTrajectoryObserver` for adding trajectories.
    """
    logger.info("Creating new Reverb replay buffer...")
    # Create the Reverb table
    table = reverb.Table(
        name=self.table_name,
        max_size=self.capacity,
        sampler=reverb.selectors.Uniform(), # Standard uniform sampling
        remover=reverb.selectors.Fifo(),   # Removes oldest data when capacity is reached
        rate_limiter=reverb.rate_limiters.MinSize(1), # Ensures some data before sampling
        # Signature matching is important for ReverbAddTrajectoryObserver
        signature=self.data_spec
    )

    # Create the checkpointer for saving/loading buffer state
    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=self.checkpoint_dir
    )

    # Create the Reverb server (in-process for simplicity here)
    # For distributed setups, this server could be external.
    reverb_server = reverb.Server(
        tables=[table], # Server manages our table
        port=None,      # `None` for in-process server, finds an available port
        checkpointer=reverb_checkpointer
    )

    # Create the TF-Agents ReplayBuffer wrapper
    replay_buffer_instance = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec=self.data_spec,
        sequence_length=self.sequence_length,
        table_name=self.table_name,
        local_server=reverb_server # Connects to our in-process server
    )

    # Create an observer for adding trajectories to the buffer
    # This observer can be passed to driver.run() or used manually.
    trajectory_observer = reverb_utils.ReverbAddTrajectoryObserver(
        py_client=replay_buffer_instance.py_client, # Client to interact with Reverb
        table_name=self.table_name,
        sequence_length=self.sequence_length,
        stride_length=1 # How many steps to slide the window for sequences
    )

    # Store components and mark as initialized
    self.server = reverb_server
    self.replay_buffer = replay_buffer_instance
    self.observer = trajectory_observer
    self._is_initialized = True
    logger.info("Reverb replay buffer created successfully.")

    return self.replay_buffer, self.observer

  def load_replay_buffer(
      self,
  ) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver,
  ]:
    """Loads an existing Reverb replay buffer from a checkpoint.

    This method attempts to restore the state of the Reverb server (and thus
    the replay buffer's data) from the `checkpoint_dir` specified during
    initialization. It then reconstructs the `ReverbReplayBuffer` and
    `ReverbAddTrajectoryObserver` to interact with the loaded server.

    Returns:
      A tuple containing:
        - replay_buffer: The loaded `ReverbReplayBuffer` instance.
        - observer: The `ReverbAddTrajectoryObserver` for adding new trajectories.

    Raises:
      tf.errors.NotFoundError: Or other Reverb/TensorFlow errors if the
        checkpoint cannot be found or loaded.
    """
    logger.info("Attempting to load Reverb replay buffer from: %s", self.checkpoint_dir)
    # Create the Reverb table (must match original configuration)
    table = reverb.Table(
        name=self.table_name,
        max_size=self.capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1), # Important for loading
        signature=self.data_spec
    )

    # Create the checkpointer
    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=self.checkpoint_dir
    )

    # Create the Reverb server, which will load from the checkpoint if available
    reverb_server = reverb.Server(
        tables=[table],
        port=None, # Let Reverb pick a port
        checkpointer=reverb_checkpointer
    )
    # Note: The actual loading happens when the server starts and the checkpointer
    # finds a valid checkpoint. Errors during this process might be raised by Reverb.

    # Create the TF-Agents ReplayBuffer wrapper
    replay_buffer_instance = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec=self.data_spec,
        sequence_length=self.sequence_length,
        table_name=self.table_name,
        local_server=reverb_server
    )

    # Create an observer for adding new trajectories
    trajectory_observer = reverb_utils.ReverbAddTrajectoryObserver(
        py_client=replay_buffer_instance.py_client,
        table_name=self.table_name,
        sequence_length=self.sequence_length,
        stride_length=1
    )

    # Store components and mark as initialized
    self.server = reverb_server
    self.replay_buffer = replay_buffer_instance
    self.observer = trajectory_observer
    self._is_initialized = True
    logger.info("Replay buffer and server components initialized for loading.")
    # At this point, Reverb server has started and attempted to load from checkpoint.
    # We can check `replay_buffer.num_frames()` to see if data was loaded.
    logger.info("Replay buffer frames after potential load: %d", self.replay_buffer.num_frames())


    return self.replay_buffer, self.observer

  def get_replay_buffer_and_observer(
      self,
  ) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver,
  ]:
    """Returns the initialized replay buffer and trajectory observer.

    If the buffer and observer have not yet been initialized (e.g., by
    `create_replay_buffer` or `load_replay_buffer`), this method will
    call `create_replay_buffer` to set them up.

    Returns:
      A tuple containing:
        - replay_buffer: The `ReverbReplayBuffer` instance.
        - observer: The `ReverbAddTrajectoryObserver` for adding trajectories.
    """
    if not self._is_initialized or self.replay_buffer is None or self.observer is None:
      logger.info("Replay buffer not initialized. Calling create_replay_buffer().")
      return self.create_replay_buffer()
    return self.replay_buffer, self.observer

  def get_dataset(
      self, batch_size: int = 64, num_steps: Optional[int] = None
  ) -> tf.data.Dataset:
    """Creates and returns a `tf.data.Dataset` for sampling from the buffer.

    This dataset can be used to provide training data to a TF-Agents agent.

    Args:
      batch_size: The number of sequences to sample in each batch from the
        replay buffer. Defaults to 64.
      num_steps: The number of timesteps to include in each sampled sequence.
        If `None`, this defaults to the `sequence_length` configured for the
        replay buffer. This is useful for n-step returns or when working with
        sequence models. Defaults to None.

    Returns:
      A `tf.data.Dataset` instance that yields batches of trajectories sampled
      from the replay buffer.

    Raises:
      RuntimeError: If the replay buffer has not been initialized by calling
        `create_replay_buffer` or `load_replay_buffer` first.
    """
    if not self._is_initialized or self.replay_buffer is None:
      raise RuntimeError(
          "Replay buffer not initialized. Call create_replay_buffer() or "
          "load_replay_buffer() before attempting to get a dataset."
      )

    # Use the replay buffer's configured sequence_length if num_steps is not specified.
    effective_num_steps = num_steps if num_steps is not None else self.sequence_length

    return self.replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=effective_num_steps
    )

  def num_frames(self) -> int:
    """Returns the total number of individual frames in the replay buffer.

    A "frame" typically corresponds to a single step in a trajectory.

    Returns:
      The current number of frames stored in the replay buffer. Returns 0 if
      the buffer is not initialized.
    """
    if not self._is_initialized or self.replay_buffer is None:
      return 0
    # num_frames() is a method on ReverbReplayBuffer that queries the server.
    return self.replay_buffer.num_frames()

  def clear(self) -> None:
    """Clears all data from the replay buffer.

    This is achieved by stopping the current Reverb server (if running) and
    then re-initializing a new, empty server and replay buffer using the
    original configuration.
    """
    if not self._is_initialized:
      logger.info("Replay buffer not initialized. Nothing to clear.")
      return

    if self.server:
      logger.info("Stopping existing Reverb server to clear buffer.")
      self.server.stop() # Ensure server is stopped before recreating.
      self.server = None # Clear reference

    logger.info("Recreating replay buffer to clear contents.")
    self.create_replay_buffer() # This re-initializes self.server, self.replay_buffer, etc.
    logger.info("Replay buffer cleared and recreated.")

  def close(self) -> None:
    """Stops the Reverb server and marks the manager as uninitialized.

    This should be called to clean up resources when the replay buffer is no
    longer needed.
    """
    if self._is_initialized and self.server:
      logger.info("Stopping Reverb replay buffer server.")
      self.server.stop()
      self.server = None
      self.replay_buffer = None
      self.observer = None
      self._is_initialized = False
      logger.info("Replay buffer server stopped and manager closed.")
    else:
      logger.info("Replay buffer already closed or not initialized.")
