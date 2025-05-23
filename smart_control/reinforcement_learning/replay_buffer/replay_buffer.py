"""Manages Reverb replay buffers for TF-Agents.

This module provides the `ReplayBufferManager` class, which encapsulates the
logic for creating, loading, saving (via checkpointing), and interacting with
Reverb-based replay buffers used in reinforcement learning.
"""

import logging
from typing import Optional, Tuple

import reverb # type: ignore[import-untyped]
import tensorflow as tf
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

logger = logging.getLogger(__name__)


class ReplayBufferManager:
  """Manages the lifecycle of a Reverb replay buffer for TF-Agents.

  This class handles the creation of a Reverb server, table, replay buffer,
  and the associated observer for adding trajectories. It also supports
  checkpointing and loading the buffer's state.

  Attributes:
    data_spec (types.NestedTensorSpec): The specification of the data that
      will be stored in the replay buffer (typically matching the agent's
      trajectory spec).
    capacity (int): The maximum number of items the replay buffer can store.
    checkpoint_dir (str): Directory where the Reverb checkpointer will save
      and load buffer states.
    sequence_length (int): The length of sequences of trajectories to store
      and sample.
    table_name (str): The name of the Reverb table used by the buffer.
    server (Optional[reverb.Server]): The Reverb server instance.
    replay_buffer (Optional[reverb_replay_buffer.ReverbReplayBuffer]): The
      TF-Agents Reverb replay buffer instance.
    observer (Optional[reverb_utils.ReverbAddTrajectoryObserver]): The observer
      used to add trajectories to the replay buffer.
    _is_initialized (bool): Flag indicating if the buffer components have been
      created or loaded.

  Example:
    ```python
    # Assuming agent_trajectory_spec is defined (e.g., from an agent)
    # trajectory_spec = tf_agents.trajectories.Trajectory(
    #    observation=tensor_spec.TensorSpec(shape=(4,), dtype=tf.float32, name='obs'),
    #    action=tensor_spec.BoundedTensorSpec(shape=(1,), dtype=tf.float32, minimum=-1, maximum=1, name='act'),
    #    policy_info=(),
    #    reward=tensor_spec.TensorSpec(shape=(), dtype=tf.float32, name='rew'),
    #    discount=tensor_spec.BoundedTensorSpec(shape=(), dtype=tf.float32, minimum=0., maximum=1., name='dsc'),
    #    step_type=tensor_spec.TensorSpec(shape=(), dtype=tf.int32, name='stp_typ'),
    #    next_step_type=tensor_spec.TensorSpec(shape=(), dtype=tf.int32, name='nxt_stp_typ')
    # )
    #
    # buffer_manager = ReplayBufferManager(
    #     data_spec=trajectory_spec,
    #     capacity=100000,
    #     checkpoint_dir="/tmp/my_replay_buffer_checkpoints"
    # )
    #
    # # Create a new buffer
    # rb, obs = buffer_manager.create_replay_buffer()
    # print(f"Buffer created. Observer: {obs}")
    #
    # # Or load an existing one if checkpoints exist
    # # rb, obs = buffer_manager.load_replay_buffer()
    #
    # # Get a dataset for training
    # dataset = buffer_manager.get_dataset(batch_size=64)
    #
    # # ... use observer to add trajectories ...
    # # driver.run(..., observers=[obs])
    #
    # print(f"Number of frames: {buffer_manager.num_frames()}")
    #
    # # Close the buffer when done
    # buffer_manager.close()
    ```
  """

  def __init__(
      self,
      data_spec: types.NestedTensorSpec,
      capacity: int,
      checkpoint_dir: str,
      sequence_length: int = 2,
  ):
    """Initializes the ReplayBufferManager.

    Args:
      data_spec (types.NestedTensorSpec): The spec of data to be stored.
        This usually corresponds to the trajectory spec of an agent.
      capacity (int): Maximum number of items (trajectories or sequences)
        the replay buffer can hold.
      checkpoint_dir (str): Path to the directory where replay buffer
        checkpoints will be saved and loaded from.
      sequence_length (int): The length of item sequences stored and sampled
        from the buffer. Defaults to 2, which is common for N-step returns or
        sequence-based models.
    """
    self.data_spec: types.NestedTensorSpec = data_spec
    self.capacity: int = capacity
    self.checkpoint_dir: str = checkpoint_dir
    self.sequence_length: int = sequence_length
    self.table_name: str = "uniform_table" # Default table name for Reverb

    self.server: Optional[reverb.Server] = None
    self.replay_buffer: Optional[reverb_replay_buffer.ReverbReplayBuffer] = None
    self.observer: Optional[reverb_utils.ReverbAddTrajectoryObserver] = None
    self._is_initialized: bool = False

  def _initialize_components(
      self, server: reverb.Server
  ) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver,
  ]:
    """Helper to initialize replay buffer and observer given a Reverb server.

    Args:
      server (reverb.Server): An initialized Reverb server instance.

    Returns:
      Tuple containing the ReverbReplayBuffer and ReverbAddTrajectoryObserver.
    """
    replay_buffer_instance = reverb_replay_buffer.ReverbReplayBuffer(
        data_spec=self.data_spec,
        sequence_length=self.sequence_length,
        table_name=self.table_name,
        local_server=server,
    )

    observer_instance = reverb_utils.ReverbAddTrajectoryObserver(
        py_client=replay_buffer_instance.py_client,
        table_name=self.table_name,
        sequence_length=self.sequence_length,
        stride_length=1, # Add every trajectory step
    )

    self.server = server
    self.replay_buffer = replay_buffer_instance
    self.observer = observer_instance
    self._is_initialized = True
    return replay_buffer_instance, observer_instance

  def create_replay_buffer(
      self,
  ) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver,
  ]:
    """Creates and initializes a new Reverb replay buffer and server.

    This involves setting up a Reverb table, a checkpointer for saving state,
    and the Reverb server itself. The replay buffer and an observer for adding
    data are then created based on this server.

    Returns:
      A tuple containing the initialized `ReverbReplayBuffer` and
      `ReverbAddTrajectoryObserver`.
    """
    table = reverb.Table(
        name=self.table_name,
        max_size=self.capacity,
        sampler=reverb.selectors.Uniform(), # Standard uniform sampling
        remover=reverb.selectors.Fifo(),    # Removes oldest items when full
        rate_limiter=reverb.rate_limiters.MinSize(1), # Wait until table has 1 item
    )

    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=self.checkpoint_dir
    )

    reverb_server = reverb.Server(
        tables=[table], port=None, checkpointer=reverb_checkpointer
    )
    logger.info("New Reverb server created for replay buffer.")
    return self._initialize_components(reverb_server)

  def load_replay_buffer(
      self,
  ) -> Tuple[
      reverb_replay_buffer.ReverbReplayBuffer,
      reverb_utils.ReverbAddTrajectoryObserver,
  ]:
    """Loads an existing replay buffer from a checkpoint.

    This method reconstructs the Reverb server, table, and checkpointer,
    allowing the server to restore its state from the specified
    `checkpoint_dir`. The replay buffer and observer are then created using
    this restored server.

    Returns:
      A tuple containing the loaded `ReverbReplayBuffer` and
      `ReverbAddTrajectoryObserver`.
    """
    table = reverb.Table(
        name=self.table_name,
        max_size=self.capacity,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
    )

    reverb_checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
        path=self.checkpoint_dir
    )

    # The server will load from the checkpointer if data exists at path.
    reverb_server = reverb.Server(
        tables=[table], port=None, checkpointer=reverb_checkpointer
    )
    logger.info(
        "Reverb server created, attempting to load from checkpoint: %s",
        self.checkpoint_dir
    )
    return self._initialize_components(reverb_server)

  def get_replay_buffer_and_observer(
      self,
  ) -> Tuple[
      Optional[reverb_replay_buffer.ReverbReplayBuffer],
      Optional[reverb_utils.ReverbAddTrajectoryObserver],
  ]:
    """Returns the managed replay buffer and observer.

    If the buffer has not been initialized (e.g., by `create_replay_buffer` or
    `load_replay_buffer`), this method will attempt to create a new one.
    Consider explicitly calling create or load for clarity.

    Returns:
      A tuple containing the `ReverbReplayBuffer` and
      `ReverbAddTrajectoryObserver`, or (None, None) if initialization fails.
    """
    if not self._is_initialized:
      logger.info(
          "Replay buffer not yet initialized. Calling create_replay_buffer()."
      )
      # This implicitly sets self.replay_buffer and self.observer
      self.create_replay_buffer()
    return self.replay_buffer, self.observer

  def get_dataset(
      self, batch_size: int = 64, num_steps: Optional[int] = None
  ) -> tf.data.Dataset:
    """Creates a `tf.data.Dataset` for sampling from the replay buffer.

    Args:
      batch_size (int): The number of sequences to sample in each batch from
        the dataset.
      num_steps (Optional[int]): The number of steps (transitions) to include
        in each sampled sequence. If None, defaults to `self.sequence_length`.

    Returns:
      tf.data.Dataset: A dataset that yields batches of trajectory sequences.

    Raises:
      RuntimeError: If the replay buffer has not been initialized.
    """
    if not self._is_initialized or self.replay_buffer is None:
      raise RuntimeError(
          "Replay buffer not initialized. Call create_replay_buffer() or "
          "load_replay_buffer() first."
      )

    # Use self.sequence_length if num_steps is not specified by the caller.
    effective_num_steps = num_steps if num_steps is not None else self.sequence_length

    return self.replay_buffer.as_dataset(
        sample_batch_size=batch_size, num_steps=effective_num_steps
    )

  def num_frames(self) -> tf.Tensor:
    """Returns the current number of frames (items) in the replay buffer.

    Returns:
      tf.Tensor: A scalar tensor representing the total number of items
      (typically individual trajectory steps or sequences based on how data is
      added) currently stored in the buffer. Returns a tensor with 0 if
      not initialized.
    """
    if not self._is_initialized or self.replay_buffer is None:
      return tf.constant(0, dtype=tf.int64)
    return self.replay_buffer.num_frames()

  def clear(self) -> None:
    """Clears all data from the replay buffer and reinitializes it.

    This stops the current Reverb server (if running) and creates a new,
    empty one with the same configuration.
    """
    if self.server:
      try:
        self.server.stop()
        logger.info("Existing Reverb server stopped for clearing.")
      except Exception as e: # pylint: disable=broad-except
        logger.error("Error stopping existing Reverb server: %s", e)

    self._is_initialized = False # Mark as uninitialized before recreation
    logger.info("Recreating replay buffer after clearing...")
    self.create_replay_buffer() # This will reinitialize all components
    logger.info("Replay buffer cleared and recreated.")

  def close(self) -> None:
    """Stops the Reverb server and marks the manager as uninitialized.

    This should be called to clean up resources when the replay buffer is no
    longer needed.
    """
    if self.server:
      try:
        self.server.stop()
        logger.info("Reverb server stopped successfully.")
      except Exception as e: # pylint: disable=broad-except
        logger.error("Error stopping Reverb server during close: %s", e)
    self.server = None
    self.replay_buffer = None
    self.observer = None
    self._is_initialized = False
    logger.info("ReplayBufferManager closed.")
