import unittest
from unittest.mock import patch, MagicMock

import tensorflow as tf

from smart_control.reinforcement_learning.replay_buffer import replay_buffer as sc_replay_buffer # aliased
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory


@patch('smart_control.reinforcement_learning.replay_buffer.replay_buffer.tf_uniform_replay_buffer.TFUniformReplayBuffer', autospec=True)
class ReplayBufferTest(tf.test.TestCase):

    def setUp(self, MockTFUniformReplayBuffer): # Mock is passed due to class decorator
        super().setUp()
        self.data_spec = trajectory.Trajectory(
            step_type=tensor_spec.TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
            observation=tensor_spec.TensorSpec(shape=(2,), dtype=tf.float32, name='observation'),
            action=tensor_spec.TensorSpec(shape=(1,), dtype=tf.float32, name='action'),
            policy_info=(),
            next_step_type=tensor_spec.TensorSpec(shape=(), dtype=tf.int32, name='next_step_type'),
            reward=tensor_spec.TensorSpec(shape=(), dtype=tf.float32, name='reward'),
            discount=tensor_spec.TensorSpec(shape=(), dtype=tf.float32, name='discount')
        )
        self.batch_size = 32
        self.max_length = 1000

        # Store the mock class from the decorator to reset it if needed between tests,
        # or use it for assertions on constructor calls if ReplayBuffer is re-instantiated.
        self.MockTFUniformReplayBuffer = MockTFUniformReplayBuffer

        # Instantiate ReplayBuffer, which will use the mocked TFUniformReplayBuffer
        self.replay_buffer = sc_replay_buffer.ReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        # Get the instance of the mocked TFUniformReplayBuffer
        self.mock_tf_buffer_instance = self.MockTFUniformReplayBuffer.return_value

    def test_init(self):
        # Assert that TFUniformReplayBuffer constructor was called correctly during setUp
        self.MockTFUniformReplayBuffer.assert_called_once_with(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        # Check if the internal tf_buffer is the mocked instance
        self.assertIs(self.replay_buffer._tf_buffer, self.mock_tf_buffer_instance)

    def _create_sample_trajectory(self):
        # Helper to create a single trajectory matching self.data_spec
        return trajectory.Trajectory(
            step_type=tf.constant(1, dtype=tf.int32),
            observation=tf.constant([1.0, 2.0], dtype=tf.float32),
            action=tf.constant([0.5], dtype=tf.float32),
            policy_info=(),
            next_step_type=tf.constant(2, dtype=tf.int32),
            reward=tf.constant(1.0, dtype=tf.float32),
            discount=tf.constant(1.0, dtype=tf.float32)
        )

    def test_add_batch(self):
        sample_traj = self._create_sample_trajectory()
        # The add_batch method expects a batch, so we need to add an outer dimension
        # to our single sample trajectory. tf_agents.utils.nest_utils.stack_nested_tensors
        # is typically used for this if you have a list of trajectories.
        # For a single trajectory to become a batch of 1:
        batched_sample_traj = tf.nest.map_structure(lambda t: tf.expand_dims(t, 0), sample_traj)

        self.replay_buffer.add_batch(batched_sample_traj)
        self.mock_tf_buffer_instance.add_batch.assert_called_once_with(batched_sample_traj)

    def test_get_next(self):
        # Configure the mock return value for the underlying buffer's get_next
        mock_sample_data = self._create_sample_trajectory() # Dummy trajectory
        # get_next returns a batch, so ensure the mock data is "batched"
        # For example, if num_steps=2, time_stacked=True, the sample data should reflect that structure.
        # Here, we simplify by assuming the mock_tf_buffer_instance handles the structure.
        # Let's assume get_next returns a tuple of (data, info)
        mock_sample_info = MagicMock(name="SampleInfoMock") 
        self.mock_tf_buffer_instance.get_next.return_value = (mock_sample_data, mock_sample_info)

        num_steps_val = 2
        time_stacked_val = True
        
        returned_data, returned_info = self.replay_buffer.get_next(
            num_steps=num_steps_val, time_stacked=time_stacked_val
        )

        self.mock_tf_buffer_instance.get_next.assert_called_once_with(
            sample_batch_size=None, # Default value from ReplayBuffer.get_next
            num_steps=num_steps_val,
            time_stacked=time_stacked_val
        )
        self.assertIs(returned_data, mock_sample_data)
        self.assertIs(returned_info, mock_sample_info)

    def test_as_dataset(self):
        mock_dataset = MagicMock(spec=tf.data.Dataset)
        self.mock_tf_buffer_instance.as_dataset.return_value = mock_dataset

        num_steps_val = 2
        num_parallel_calls_val = 3
        single_deterministic_pass_val = True

        dataset_result = self.replay_buffer.as_dataset(
            num_steps=num_steps_val,
            num_parallel_calls=num_parallel_calls_val,
            single_deterministic_pass=single_deterministic_pass_val
        )

        self.mock_tf_buffer_instance.as_dataset.assert_called_once_with(
            sample_batch_size=None, # Default from ReplayBuffer.as_dataset
            num_steps=num_steps_val,
            num_parallel_calls=num_parallel_calls_val,
            single_deterministic_pass=single_deterministic_pass_val
        )
        self.assertIs(dataset_result, mock_dataset)

    def test_gather_all(self):
        mock_all_trajectories = self._create_sample_trajectory() # Create a dummy trajectory
        self.mock_tf_buffer_instance.gather_all.return_value = mock_all_trajectories

        gathered_trajectories = self.replay_buffer.gather_all()

        self.mock_tf_buffer_instance.gather_all.assert_called_once_with()
        self.assertIs(gathered_trajectories, mock_all_trajectories)

    def test_clear(self):
        self.replay_buffer.clear()
        self.mock_tf_buffer_instance.clear.assert_called_once_with()

    def test_data_spec_property(self):
        # Configure the mock underlying buffer's data_spec property
        # (which is what our ReplayBuffer.data_spec should return)
        self.mock_tf_buffer_instance.data_spec = self.data_spec
        
        retrieved_data_spec = self.replay_buffer.data_spec
        self.assertIs(retrieved_data_spec, self.data_spec)
        # No direct call to a method like "get_data_spec", so we check property access.
        # If data_spec is a simple attribute on the mock, this is fine.
        # If it's a property on TFUniformReplayBuffer, the mock should replicate that.
        # autospec=True helps in making the mock behave more like the original.


if __name__ == '__main__':
    tf.test.main()
