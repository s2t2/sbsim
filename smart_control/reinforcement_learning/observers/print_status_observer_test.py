import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import numpy as np
import tensorflow as tf

from smart_control.reinforcement_learning.observers import print_status_observer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step


class PrintStatusObserverTest(tf.test.TestCase):

    def _create_dummy_trajectory(self, step_type_val, next_step_type_val, observation_val, action_val, reward_val, discount_val, batched=False, batch_size=1):
        if batched:
            step_type = np.array([step_type_val] * batch_size, dtype=np.int32)
            observation = np.array([observation_val] * batch_size, dtype=np.float32)
            action = np.array([action_val] * batch_size, dtype=np.float32)
            policy_info = () # Or make it batched if necessary for your tests
            next_step_type = np.array([next_step_type_val] * batch_size, dtype=np.int32)
            reward = np.array([reward_val] * batch_size, dtype=np.float32)
            discount = np.array([discount_val] * batch_size, dtype=np.float32)
        else:
            step_type = np.array(step_type_val, dtype=np.int32)
            observation = np.array(observation_val, dtype=np.float32)
            action = np.array(action_val, dtype=np.float32)
            policy_info = ()
            next_step_type = np.array(next_step_type_val, dtype=np.int32)
            reward = np.array(reward_val, dtype=np.float32)
            discount = np.array(discount_val, dtype=np.float32)

        return trajectory.Trajectory(
            step_type=step_type,
            observation=observation,
            action=action,
            policy_info=policy_info,
            next_step_type=next_step_type,
            reward=reward,
            discount=discount
        )

    def test_init_with_print_fn_and_priority(self):
        mock_print = MagicMock()
        priority = 10
        observer = print_status_observer.PrintStatusObserver(print_fn=mock_print, priority=priority)
        self.assertIs(observer._print_fn, mock_print)
        self.assertEqual(observer.priority, priority)

    def test_init_default_priority(self):
        mock_print = MagicMock()
        observer = print_status_observer.PrintStatusObserver(print_fn=mock_print)
        self.assertEqual(observer.priority, 0) # Assuming default is 0 as per BaseObserver

    def test_init_default_print_fn(self):
        # Test that it defaults to built-in print if no print_fn is provided
        observer = print_status_observer.PrintStatusObserver()
        # Not easy to directly assert it's the built-in print,
        # but we can check it's callable and doesn't fail during call.
        self.assertTrue(callable(observer._print_fn))
        # Further test of default print_fn will happen in __call__ tests using stdout capture

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_call_single_trajectory_mid_step_default_print(self, mock_stdout):
        observer = print_status_observer.PrintStatusObserver() # Uses default print

        obs_val = [1.0, 2.0]
        act_val = [0.5]
        reward_val = 10.0
        discount_val = 0.9

        traj = self._create_dummy_trajectory(
            step_type_val=time_step.StepType.MID,
            next_step_type_val=time_step.StepType.LAST,
            observation_val=obs_val,
            action_val=act_val,
            reward_val=reward_val,
            discount_val=discount_val
        )
        observer(traj)
        output = mock_stdout.getvalue()

        self.assertIn("Step Type: MID", output)
        self.assertIn(f"Observation: {np.array(obs_val, dtype=np.float32)}", output)
        self.assertIn(f"Action: {np.array(act_val, dtype=np.float32)}", output)
        self.assertIn(f"Reward: {reward_val}", output)
        self.assertIn(f"Discount: {discount_val}", output)
        self.assertIn("Next Step Type: LAST", output)
        self.assertIn("Policy Info: ()", output) # Assuming default policy_info is empty tuple

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_call_single_trajectory_first_step(self, mock_stdout):
        observer = print_status_observer.PrintStatusObserver()
        traj = self._create_dummy_trajectory(
            step_type_val=time_step.StepType.FIRST,
            next_step_type_val=time_step.StepType.MID,
            observation_val=[0.,0.], action_val=[0.], reward_val=0., discount_val=1.
        )
        observer(traj)
        output = mock_stdout.getvalue()
        self.assertIn("Step Type: FIRST", output)
        self.assertIn("Reward: 0.0", output) # Reward is typically 0 for FIRST step

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_call_single_trajectory_last_step(self, mock_stdout):
        observer = print_status_observer.PrintStatusObserver()
        final_reward = 50.0
        traj = self._create_dummy_trajectory(
            step_type_val=time_step.StepType.LAST,
            next_step_type_val=time_step.StepType.FIRST, # Arbitrary for this test, usually LAST has specific next_step_type
            observation_val=[10.,10.], action_val=[1.], reward_val=final_reward, discount_val=0.
        )
        observer(traj)
        output = mock_stdout.getvalue()
        self.assertIn("Step Type: LAST", output)
        self.assertIn(f"Reward: {final_reward}", output)
        self.assertIn("Discount: 0.0", output) # Discount is typically 0 for LAST step

    def test_call_with_custom_print_fn(self):
        mock_custom_print = MagicMock()
        observer = print_status_observer.PrintStatusObserver(print_fn=mock_custom_print)
        traj = self._create_dummy_trajectory(
            step_type_val=time_step.StepType.MID,
            next_step_type_val=time_step.StepType.LAST,
            observation_val=[1.,2.], action_val=[0.5], reward_val=10., discount_val=0.9
        )
        observer(traj)

        mock_custom_print.assert_called_once()
        call_args_str = mock_custom_print.call_args[0][0] # Get the string passed to print
        self.assertIn("Step Type: MID", call_args_str)
        self.assertIn("Observation: [1. 2.]", call_args_str)
        self.assertIn("Action: [0.5]", call_args_str)
        self.assertIn("Reward: 10.0", call_args_str)

    @patch('sys.stdout', new_callable=io.StringIO)
    def test_call_batched_trajectory_default_print(self, mock_stdout):
        observer = print_status_observer.PrintStatusObserver()

        batch_size = 2
        obs_val = [1.0, 2.0]
        act_val = [0.5]
        reward_val1 = 10.0
        reward_val2 = 20.0
        discount_val = 0.9

        # Create a batched trajectory
        # Note: The _create_dummy_trajectory can be used with batched=True
        traj = self._create_dummy_trajectory(
            step_type_val=time_step.StepType.MID, # Same step_type for both items in batch
            next_step_type_val=time_step.StepType.LAST,
            observation_val=obs_val, # Same obs for both for simplicity
            action_val=act_val,      # Same action
            reward_val=reward_val1,  # This will be used for all in current helper, need to adjust if different rewards
            discount_val=discount_val,
            batched=True,
            batch_size=batch_size
        )
        # Manually adjust rewards for each item in the batch if _create_dummy_trajectory doesn't support it
        # For this test, let's assume the observer prints each item.
        # If the PrintStatusObserver's __call__ iterates, it will process each trajectory in the batch.
        # Let's make a trajectory where rewards differ to check individual printing
        traj_item1 = self._create_dummy_trajectory(
            step_type_val=time_step.StepType.MID, next_step_type_val=time_step.StepType.LAST,
            observation_val=[1.,2.], action_val=[0.1], reward_val=11., discount_val=0.99
        )
        traj_item2 = self._create_dummy_trajectory(
            step_type_val=time_step.StepType.FIRST, next_step_type_val=time_step.StepType.MID,
            observation_val=[3.,4.], action_val=[0.2], reward_val=22., discount_val=0.98
        )

        # Stack them to create a batch
        batched_traj = tf.nest.map_structure(lambda *x: np.stack(x), traj_item1, traj_item2)


        observer(batched_traj)
        output = mock_stdout.getvalue()

        # Check for output from the first trajectory in the batch
        self.assertIn("Step Type: MID", output) # From traj_item1
        self.assertIn("Observation: [1. 2.]", output)
        self.assertIn("Action: [0.1]", output)
        self.assertIn("Reward: 11.0", output)
        self.assertIn("Discount: 0.99", output)
        self.assertIn("Next Step Type: LAST", output)

        # Check for output from the second trajectory in the batch
        self.assertIn("Step Type: FIRST", output) # From traj_item2
        self.assertIn("Observation: [3. 4.]", output)
        self.assertIn("Action: [0.2]", output)
        self.assertIn("Reward: 22.0", output)
        self.assertIn("Discount: 0.98", output)
        self.assertIn("Next Step Type: MID", output)

        # Verify that print was called twice (once for each item in the batch)
        # This can be inferred if the output contains distinct parts of both trajectories.
        # Or, if using a mock_print_fn, check call_count.
        # For stdout capture, we check for presence of both items' data.
        num_separators = output.count("--- Trajectory ---") # Assuming the observer prints a separator
        if num_separators > 0 : # Only assert if separator is used
             self.assertEqual(num_separators, batch_size)


if __name__ == '__main__':
    tf.test.main()
