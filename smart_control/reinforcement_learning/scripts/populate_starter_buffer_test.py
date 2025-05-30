import unittest
from unittest.mock import patch, MagicMock, call

import tensorflow as tf
import numpy as np

from absl import flags
# Import the script to be tested
from smart_control.reinforcement_learning.scripts import populate_starter_buffer

from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from smart_control.reinforcement_learning.replay_buffer import replay_buffer as sc_replay_buffer # aliased
from tf_agents.trajectories import trajectory
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import policy_step


class PopulateStarterBufferTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        # Define common specs for environment and policy
        self.observation_spec = tensor_spec.TensorSpec((2,), tf.float32, 'obs')
        self.action_spec = tensor_spec.BoundedTensorSpec((1,), tf.float32, minimum=0, maximum=1, name='act')
        self.time_step_spec = ts.time_step_spec(self.observation_spec)

        # Mock FLAGS that might be accessed by the function or its callees
        # If populate_buffer_fn directly uses FLAGS, mock them here.
        # For now, assume it takes parameters directly.
        # FLAGS = flags.FLAGS
        # FLAGS(['test_program', '--root_dir=test_dir']) # Example if FLAGS were used

    @patch('smart_control.reinforcement_learning.scripts.populate_starter_buffer.gin') # Mock gin if it's used directly in populate_buffer_fn
    @patch.object(sc_replay_buffer, 'ReplayBuffer', autospec=True) # Mock ReplayBuffer class
    @patch.object(tf_py_environment, 'TFPyEnvironment', autospec=True) # Mock TFPyEnvironment class
    @patch.object(tf_policy, 'TFPolicy', autospec=True) # Mock TFPolicy (base class for SchedulePolicy)
    def test_populate_buffer_fn_basic_loop(self, MockTFPolicy, MockTFPyEnvironment, MockReplayBuffer, MockGin):
        # --- Setup Mocks ---
        mock_env_instance = MockTFPyEnvironment.return_value
        mock_env_instance.time_step_spec.return_value = self.time_step_spec
        mock_env_instance.action_spec.return_value = self.action_spec

        mock_policy_instance = MockTFPolicy.return_value # This would be the SchedulePolicy instance
        # Configure policy's action_spec and time_step_spec if they are accessed
        mock_policy_instance.action_spec = self.action_spec
        mock_policy_instance.time_step_spec = self.time_step_spec


        mock_replay_buffer_instance = MockReplayBuffer.return_value

        # --- Control Environment and Policy Behavior ---
        # Initial time step
        initial_obs = np.array([1.0, 1.0], dtype=np.float32)
        initial_time_step = ts.restart(initial_obs, batch_size=1)
        mock_env_instance.current_time_step.return_value = initial_time_step

        # Policy action
        action_values = [tf.constant([[0.1 + i/10]], dtype=tf.float32) for i in range(5)]
        policy_actions = [policy_step.PolicyStep(action=act) for act in action_values]
        mock_policy_instance.action.side_effect = policy_actions

        # Environment step results
        next_obs_values = [np.array([1.0 + (i+1)/10, 1.0 + (i+1)/10], dtype=np.float32) for i in range(5)]
        next_time_steps = [ts.transition(next_obs_values[i], reward=tf.constant([1.0], dtype=tf.float32), discount=tf.constant([1.0], dtype=tf.float32)) for i in range(4)]
        # Make the last step a terminal one
        final_time_step = ts.termination(next_obs_values[4], reward=tf.constant([1.0], dtype=tf.float32))
        mock_env_instance.step.side_effect = next_time_steps + [final_time_step]

        # Configure replay buffer's data_spec (important for trajectory.from_transition)
        # This spec should match the trajectory that from_transition will create.
        # It is derived from the time_step_spec and policy_step_spec (which is action_spec here).
        # For simplicity, let's assume the test ensures compatible trajectories are added.
        # A more robust mock would have data_spec configured on the mock_replay_buffer_instance.
        # For example:
        # sample_traj = trajectory.from_transition(initial_time_step, policy_actions[0], next_time_steps[0])
        # mock_replay_buffer_instance.data_spec = tensor_spec.from_spec(array_spec.ArraySpec.from_tensor_spec(sample_traj.data_spec))


        # --- Call the function under test ---
        num_steps_to_collect = 5
        root_dir_val = 'mock_test_dir'
        replay_buffer_save_dir_val = 'test_replay_buffer'

        populate_starter_buffer.populate_buffer_fn(
            environment=mock_env_instance,
            policy=mock_policy_instance,
            replay_buffer=mock_replay_buffer_instance,
            root_dir=root_dir_val,
            num_steps=num_steps_to_collect,
            start_step=0, # Default
            replay_buffer_save_dir=replay_buffer_save_dir_val
        )

        # --- Verify Interactions ---
        self.assertEqual(mock_env_instance.current_time_step.call_count, 1) # Called once at the beginning
        self.assertEqual(mock_policy_instance.action.call_count, num_steps_to_collect)
        self.assertEqual(mock_env_instance.step.call_count, num_steps_to_collect)
        self.assertEqual(mock_replay_buffer_instance.add_batch.call_count, num_steps_to_collect)

        # Check arguments for replay_buffer.add_batch
        # The actual trajectory added will be trajectory.from_transition(time_step, policy_step, next_time_step)
        current_ts = initial_time_step
        for i in range(num_steps_to_collect):
            expected_policy_step = policy_actions[i]
            expected_next_ts = mock_env_instance.step.side_effect[i]
            expected_traj = trajectory.from_transition(current_ts, expected_policy_step, expected_next_ts)
            
            # Compare relevant fields of the trajectory if direct object comparison is tricky due to new object creation
            actual_call_args = mock_replay_buffer_instance.add_batch.call_args_list[i][0][0] # first arg of ith call
            self.assertAllEqual(actual_call_args.observation, expected_traj.observation)
            self.assertAllEqual(actual_call_args.action, expected_traj.action)
            self.assertAllEqual(actual_call_args.reward, expected_traj.reward)
            self.assertAllEqual(actual_call_args.step_type, expected_traj.step_type)
            self.assertAllEqual(actual_call_args.next_step_type, expected_traj.next_step_type)
            self.assertAllEqual(actual_call_args.discount, expected_traj.discount)
            current_ts = expected_next_ts


        mock_replay_buffer_instance.save.assert_called_once_with(f"{root_dir_val}/{replay_buffer_save_dir_val}")

    @patch('smart_control.reinforcement_learning.scripts.populate_starter_buffer.gin')
    @patch.object(sc_replay_buffer, 'ReplayBuffer', autospec=True)
    @patch.object(tf_py_environment, 'TFPyEnvironment', autospec=True)
    @patch.object(tf_policy, 'TFPolicy', autospec=True)
    def test_populate_buffer_fn_with_start_step(self, MockTFPolicy, MockTFPyEnvironment, MockReplayBuffer, MockGin):
        mock_env_instance = MockTFPyEnvironment.return_value
        mock_env_instance.time_step_spec.return_value = self.time_step_spec
        mock_env_instance.action_spec.return_value = self.action_spec

        mock_policy_instance = MockTFPolicy.return_value
        mock_policy_instance.action_spec = self.action_spec
        mock_policy_instance.time_step_spec = self.time_step_spec
        # Mock the train_step_counter if it's a TF Variable or similar
        mock_policy_instance.train_step_counter = tf.Variable(0, dtype=tf.int64, name="train_step_counter")


        mock_replay_buffer_instance = MockReplayBuffer.return_value

        initial_obs = np.array([1.0, 1.0], dtype=np.float32)
        initial_time_step = ts.restart(initial_obs, batch_size=1)
        mock_env_instance.current_time_step.return_value = initial_time_step

        action_val = tf.constant([[0.1]], dtype=tf.float32)
        policy_action = policy_step.PolicyStep(action=action_val)
        mock_policy_instance.action.return_value = policy_action # Simple action for all steps

        next_obs_val = np.array([1.1, 1.1], dtype=np.float32)
        next_time_step_val = ts.transition(next_obs_val, reward=tf.constant([1.0], dtype=tf.float32), discount=tf.constant([1.0], dtype=tf.float32))
        mock_env_instance.step.return_value = next_time_step_val # Simple next_time_step for all steps

        num_steps_to_collect = 3
        start_step_val = 50 # Non-zero start_step
        root_dir_val = 'mock_test_dir_start_step'
        replay_buffer_save_dir_val = 'test_replay_buffer_start_step'

        populate_starter_buffer.populate_buffer_fn(
            environment=mock_env_instance,
            policy=mock_policy_instance,
            replay_buffer=mock_replay_buffer_instance,
            root_dir=root_dir_val,
            num_steps=num_steps_to_collect,
            start_step=start_step_val,
            replay_buffer_save_dir=replay_buffer_save_dir_val
        )

        self.assertEqual(mock_policy_instance.action.call_count, num_steps_to_collect)
        self.assertEqual(mock_replay_buffer_instance.add_batch.call_count, num_steps_to_collect)
        mock_replay_buffer_instance.save.assert_called_once_with(f"{root_dir_val}/{replay_buffer_save_dir_val}")
        
        # Verify that the policy's train_step_counter was updated
        # The function is expected to call `assign` on the train_step_counter
        # If the train_step_counter is a MagicMock wrapping a tf.Variable, direct check of .numpy() or .assign call is possible.
        # Here, we mocked it as a real tf.Variable.
        # The function populate_buffer_fn calls policy.update_time_step(time_step.step_type)
        # which in turn updates the train_step_counter.
        # However, the provided populate_buffer_fn doesn't directly call `policy.update_time_step`.
        # It calls `common.function(policy.train_step_counter.assign)(time_step.step_type)`
        # This is unusual. Let's assume the intention is to increment it.
        # The test for this specific counter update logic would be more complex as it's inside common.function.
        # For now, we assume the main purpose of start_step is for logging or potential future use,
        # as the current populate_buffer_fn does not seem to use policy.train_step_counter directly.
        # The provided code actually uses `policy.train_step_counter.assign(start_step + i + 1)`
        # So, after the loop, it should be start_step + num_steps_to_collect
        self.assertEqual(mock_policy_instance.train_step_counter.numpy(), start_step_val + num_steps_to_collect)


if __name__ == '__main__':
    tf.test.main()
