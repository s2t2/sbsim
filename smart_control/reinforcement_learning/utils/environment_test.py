import unittest
from unittest.mock import patch, MagicMock, call

import tensorflow as tf
import numpy as np

from smart_control.reinforcement_learning.utils import environment as env_utils
from tf_agents.environments import suite_gym # Mocked
from tf_agents.environments import tf_py_environment # Partially mocked (constructor)
from tf_agents.environments import wrappers # Mocked
from tf_agents.policies import tf_policy # For spec
from tf_agents.trajectories import time_step as ts_lib # Renamed to avoid conflict
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import trajectory # For trajectory.Trajectory
from tf_agents.specs import array_spec, tensor_spec


class EnvironmentUtilsTest(tf.test.TestCase):

    @patch('tf_agents.environments.tf_py_environment.TFPyEnvironment', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionRepeat', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionDiscretizeWrapper', autospec=True)
    @patch('tf_agents.environments.suite_gym.load', autospec=True)
    def test_create_tf_environment_basic(self, mock_suite_gym_load, MockActionDiscretizeWrapper, MockActionRepeat, MockTFPyEnvironment):
        env_name = 'CartPole-v0'
        mock_gym_env_loaded = MagicMock()
        mock_suite_gym_load.return_value = mock_gym_env_loaded

        created_env = env_utils.create_tf_environment(env_name)

        mock_suite_gym_load.assert_called_once_with(env_name)
        MockTFPyEnvironment.assert_called_once_with(mock_gym_env_loaded)
        MockActionDiscretizeWrapper.assert_not_called()
        MockActionRepeat.assert_not_called()
        self.assertIs(created_env, MockTFPyEnvironment.return_value)

    @patch('tf_agents.environments.tf_py_environment.TFPyEnvironment', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionRepeat', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionDiscretizeWrapper', autospec=True)
    @patch('tf_agents.environments.suite_gym.load', autospec=True)
    def test_create_tf_environment_with_control_freq_and_action_bins(self, mock_suite_gym_load, MockActionDiscretizeWrapper, MockActionRepeat, MockTFPyEnvironment):
        env_name = 'MyEnv-v0'
        control_frequency = 5
        action_bins = [np.array([-1.0, 0.0, 1.0])]
        
        mock_gym_env_loaded = MagicMock(name="GymEnvLoaded")
        mock_suite_gym_load.return_value = mock_gym_env_loaded

        mock_discretized_env = MagicMock(name="DiscretizedEnv")
        MockActionDiscretizeWrapper.return_value = mock_discretized_env

        mock_repeated_env = MagicMock(name="RepeatedEnv")
        MockActionRepeat.return_value = mock_repeated_env
        
        created_env = env_utils.create_tf_environment(
            env_name,
            control_frequency=control_frequency,
            action_bins=action_bins
        )

        mock_suite_gym_load.assert_called_once_with(env_name)
        MockActionDiscretizeWrapper.assert_called_once_with(mock_gym_env_loaded, action_bins)
        MockActionRepeat.assert_called_once_with(mock_discretized_env, control_frequency)
        MockTFPyEnvironment.assert_called_once_with(mock_repeated_env)
        self.assertIs(created_env, MockTFPyEnvironment.return_value)

    @patch('tf_agents.environments.tf_py_environment.TFPyEnvironment', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionRepeat', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionDiscretizeWrapper', autospec=True)
    @patch('tf_agents.environments.suite_gym.load', autospec=True)
    def test_create_tf_environment_only_control_freq(self, mock_suite_gym_load, MockActionDiscretizeWrapper, MockActionRepeat, MockTFPyEnvironment):
        env_name = 'MyEnvFreq-v0'
        control_frequency = 3

        mock_gym_env_loaded = MagicMock(name="GymEnvLoadedFreq")
        mock_suite_gym_load.return_value = mock_gym_env_loaded
        
        mock_repeated_env = MagicMock(name="RepeatedEnvFreq")
        MockActionRepeat.return_value = mock_repeated_env

        created_env = env_utils.create_tf_environment(
            env_name,
            control_frequency=control_frequency
        )

        mock_suite_gym_load.assert_called_once_with(env_name)
        MockActionDiscretizeWrapper.assert_not_called()
        MockActionRepeat.assert_called_once_with(mock_gym_env_loaded, control_frequency) # Called with the original gym env
        MockTFPyEnvironment.assert_called_once_with(mock_repeated_env)
        self.assertIs(created_env, MockTFPyEnvironment.return_value)

    @patch('tf_agents.environments.tf_py_environment.TFPyEnvironment', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionRepeat', autospec=True)
    @patch('tf_agents.environments.wrappers.ActionDiscretizeWrapper', autospec=True)
    @patch('tf_agents.environments.suite_gym.load', autospec=True)
    def test_create_tf_environment_only_action_bins(self, mock_suite_gym_load, MockActionDiscretizeWrapper, MockActionRepeat, MockTFPyEnvironment):
        env_name = 'MyEnvBins-v0'
        action_bins = [np.array([-0.5, 0.5])]

        mock_gym_env_loaded = MagicMock(name="GymEnvLoadedBins")
        mock_suite_gym_load.return_value = mock_gym_env_loaded

        mock_discretized_env = MagicMock(name="DiscretizedEnvBins")
        MockActionDiscretizeWrapper.return_value = mock_discretized_env

        created_env = env_utils.create_tf_environment(
            env_name,
            action_bins=action_bins
        )

        mock_suite_gym_load.assert_called_once_with(env_name)
        MockActionDiscretizeWrapper.assert_called_once_with(mock_gym_env_loaded, action_bins)
        MockActionRepeat.assert_not_called()
        MockTFPyEnvironment.assert_called_once_with(mock_discretized_env) # Called with the discretized env
        self.assertIs(created_env, MockTFPyEnvironment.return_value)

    def _setup_observe_mocks(self):
        mock_tf_env = MagicMock(spec=tf_py_environment.TFPyEnvironment)
        mock_policy = MagicMock(spec=tf_policy.TFPolicy)
        
        obs_spec = tensor_spec.TensorSpec((2,), tf.float32, 'obs')
        action_spec = tensor_spec.BoundedTensorSpec((1,), tf.float32, minimum=0, maximum=1, name='act')
        time_step_spec = ts_lib.time_step_spec(obs_spec)

        mock_tf_env.time_step_spec.return_value = time_step_spec
        mock_tf_env.action_spec.return_value = action_spec
        mock_tf_env.batch_size = 1 # Important for trajectory creation
        mock_tf_env.observation_spec.return_value = obs_spec # if accessed directly
        
        # Policy also needs specs for trajectory.from_transition
        mock_policy.action_spec = action_spec
        mock_policy.time_step_spec = time_step_spec
        mock_policy.trajectory_spec = trajectory.Trajectory( # if accessed directly
            step_type=time_step_spec.step_type,
            observation=time_step_spec.observation,
            action=action_spec,
            policy_info=(), # Assuming empty policy_info for mock
            next_step_type=time_step_spec.step_type,
            reward=time_step_spec.reward,
            discount=time_step_spec.discount
        )


        mock_observer1 = MagicMock(spec=callable) # Simple callable mock
        mock_observer2 = MagicMock(spec=callable)

        return mock_tf_env, mock_policy, mock_observer1, mock_observer2, obs_spec, action_spec

    def test_observe_tf_environment_num_episodes(self):
        mock_tf_env, mock_policy, mock_observer1, mock_observer2, obs_spec, action_spec = self._setup_observe_mocks()

        num_episodes_to_run = 2
        steps_per_episode = 3 # Define how many MID steps before a LAST step

        # --- Configure mock behaviors ---
        # env.reset()
        initial_obs_ep1 = np.array([0.1, 0.1], dtype=np.float32)
        initial_obs_ep2 = np.array([0.2, 0.2], dtype=np.float32)
        mock_tf_env.reset.side_effect = [
            ts_lib.restart(initial_obs_ep1, batch_size=1),
            ts_lib.restart(initial_obs_ep2, batch_size=1)
        ]
        # For current_time_step if reset_env=False (not used in this test directly, but good practice)
        mock_tf_env.current_time_step.return_value = ts_lib.restart(initial_obs_ep1, batch_size=1)


        # policy.action()
        mock_actions = [
            policy_step.PolicyStep(action=tf.constant([[0.1 + i * 0.01]], dtype=tf.float32)) for i in range(num_episodes_to_run * steps_per_episode)
        ]
        mock_policy.action.side_effect = mock_actions

        # env.step() - Simulate episodes
        env_step_side_effects = []
        for ep in range(num_episodes_to_run):
            for step in range(steps_per_episode -1): # MID steps
                env_step_side_effects.append(ts_lib.transition(
                    observation=np.array([0.1 + ep + (step+1)*0.01, 0.1 + ep + (step+1)*0.01], dtype=np.float32),
                    reward=tf.constant([1.0], dtype=tf.float32),
                    discount=tf.constant([1.0], dtype=tf.float32)
                ))
            # LAST step for the episode
            env_step_side_effects.append(ts_lib.termination(
                observation=np.array([0.1 + ep + steps_per_episode*0.01, 0.1 + ep + steps_per_episode*0.01], dtype=np.float32),
                reward=tf.constant([1.0], dtype=tf.float32)
            ))
        mock_tf_env.step.side_effect = env_step_side_effects
        
        # --- Call the function ---
        env_utils.observe_tf_environment(
            mock_tf_env,
            mock_policy,
            observers=[mock_observer1, mock_observer2],
            num_episodes=num_episodes_to_run
        )

        # --- Assertions ---
        self.assertEqual(mock_tf_env.reset.call_count, num_episodes_to_run)
        total_steps_taken = num_episodes_to_run * steps_per_episode
        self.assertEqual(mock_policy.action.call_count, total_steps_taken)
        self.assertEqual(mock_tf_env.step.call_count, total_steps_taken)
        self.assertEqual(mock_observer1.call_count, total_steps_taken)
        self.assertEqual(mock_observer2.call_count, total_steps_taken)

        # Verify trajectory passed to observers (check first call for observer1)
        first_call_args = mock_observer1.call_args_list[0][0]
        traj_arg = first_call_args[0]
        self.assertIsInstance(traj_arg, trajectory.Trajectory)
        self.assertAllEqual(traj_arg.observation.numpy(), initial_obs_ep1.reshape(1,2)) # Reshape for batch dim
        self.assertAllEqual(traj_arg.action.numpy(), mock_actions[0].action.numpy())
        self.assertTrue(traj_arg.is_first())

    def test_observe_tf_environment_num_steps(self):
        mock_tf_env, mock_policy, mock_observer1, mock_observer2, obs_spec, action_spec = self._setup_observe_mocks()
        num_steps_to_run = 10

        # Configure mock behaviors
        initial_obs = np.array([0.5, 0.5], dtype=np.float32)
        # env.reset() will be called once at the beginning (default behavior)
        mock_tf_env.reset.return_value = ts_lib.restart(initial_obs, batch_size=1)
        # current_time_step is used if reset_env=False, but also internally after reset
        mock_tf_env.current_time_step.return_value = ts_lib.restart(initial_obs, batch_size=1)


        mock_actions = [
            policy_step.PolicyStep(action=tf.constant([[0.2 + i * 0.01]], dtype=tf.float32)) for i in range(num_steps_to_run)
        ]
        mock_policy.action.side_effect = mock_actions

        env_step_side_effects = []
        for i in range(num_steps_to_run):
            # Make one of the steps a LAST step to ensure reset logic is hit if num_steps > episode length
            if i == 5: # Let's say episode ends after 6 steps (0 to 5)
                 env_step_side_effects.append(ts_lib.termination(
                    observation=np.array([0.6, 0.6], dtype=np.float32),
                    reward=tf.constant([1.0], dtype=tf.float32)
                ))
            else:
                env_step_side_effects.append(ts_lib.transition(
                    observation=np.array([0.5 + (i+1)*0.01, 0.5 + (i+1)*0.01], dtype=np.float32),
                    reward=tf.constant([1.0], dtype=tf.float32),
                    discount=tf.constant([1.0], dtype=tf.float32)
                ))
        mock_tf_env.step.side_effect = env_step_side_effects

        env_utils.observe_tf_environment(
            mock_tf_env,
            mock_policy,
            observers=[mock_observer1], # Only one observer for simplicity
            num_steps=num_steps_to_run,
            num_episodes=0 # Explicitly set to 0 or rely on default
        )

        # Assertions
        # Reset is called once at the beginning, and once after the episode ends at step 6
        self.assertEqual(mock_tf_env.reset.call_count, 2)
        self.assertEqual(mock_policy.action.call_count, num_steps_to_run)
        self.assertEqual(mock_tf_env.step.call_count, num_steps_to_run)
        self.assertEqual(mock_observer1.call_count, num_steps_to_run)

    def test_observe_tf_environment_no_observers(self):
        mock_tf_env, mock_policy, _, _, obs_spec, action_spec = self._setup_observe_mocks()
        num_steps_to_run = 3

        initial_obs = np.array([0.0, 0.0], dtype=np.float32)
        mock_tf_env.reset.return_value = ts_lib.restart(initial_obs, batch_size=1)
        mock_tf_env.current_time_step.return_value = ts_lib.restart(initial_obs, batch_size=1)


        mock_policy.action.return_value = policy_step.PolicyStep(action=tf.constant([[0.0]], dtype=tf.float32))
        mock_tf_env.step.return_value = ts_lib.transition(initial_obs, tf.constant([1.0]), tf.constant([1.0]))

        try:
            env_utils.observe_tf_environment(
                mock_tf_env,
                mock_policy,
                observers=None, # No observers
                num_steps=num_steps_to_run
            )
        except Exception as e:
            self.fail(f"observe_tf_environment failed with no observers: {e}")

        self.assertEqual(mock_tf_env.reset.call_count, 1) # Resets once at the beginning.
        self.assertEqual(mock_policy.action.call_count, num_steps_to_run)
        self.assertEqual(mock_tf_env.step.call_count, num_steps_to_run)

    def test_observe_tf_environment_reset_env_false(self):
        mock_tf_env, mock_policy, mock_observer1, _, obs_spec, action_spec = self._setup_observe_mocks()
        num_steps_to_run = 5 # Run for a few steps

        # Configure current_time_step as reset_env=False will use this
        initial_obs = np.array([0.7, 0.7], dtype=np.float32)
        # Simulate that the first time_step is MID (not FIRST)
        current_ts_val = ts_lib.transition(initial_obs, reward=tf.constant([0.0], dtype=tf.float32), discount=tf.constant([1.0],dtype=tf.float32))
        mock_tf_env.current_time_step.return_value = current_ts_val

        mock_actions = [
            policy_step.PolicyStep(action=tf.constant([[0.3 + i * 0.01]], dtype=tf.float32)) for i in range(num_steps_to_run)
        ]
        mock_policy.action.side_effect = mock_actions

        env_step_side_effects = []
        for i in range(num_steps_to_run):
            if i == 2: # Episode ends at 3rd step (index 2)
                env_step_side_effects.append(ts_lib.termination(
                    observation=np.array([0.8, 0.8], dtype=np.float32),
                    reward=tf.constant([1.0], dtype=tf.float32)
                ))
            else:
                env_step_side_effects.append(ts_lib.transition(
                    observation=np.array([0.7 + (i+1)*0.01, 0.7 + (i+1)*0.01], dtype=np.float32),
                    reward=tf.constant([1.0], dtype=tf.float32),
                    discount=tf.constant([1.0], dtype=tf.float32)
                ))
        mock_tf_env.step.side_effect = env_step_side_effects
        
        # Mock reset for when the episode actually ends
        mock_tf_env.reset.return_value = ts_lib.restart(np.array([0.9,0.9], dtype=np.float32), batch_size=1)


        env_utils.observe_tf_environment(
            mock_tf_env,
            mock_policy,
            observers=[mock_observer1],
            num_steps=num_steps_to_run,
            reset_env=False # Key parameter for this test
        )

        # Assertions
        mock_tf_env.current_time_step.assert_called_once() # Called at the beginning
        mock_tf_env.reset.assert_called_once() # Called after the episode ended at step 3
        
        self.assertEqual(mock_policy.action.call_count, num_steps_to_run)
        self.assertEqual(mock_tf_env.step.call_count, num_steps_to_run)
        self.assertEqual(mock_observer1.call_count, num_steps_to_run)

        # Check that the first trajectory passed to observer was based on the initial current_ts_val
        first_call_args = mock_observer1.call_args_list[0][0]
        traj_arg = first_call_args[0]
        self.assertIsInstance(traj_arg, trajectory.Trajectory)
        self.assertAllEqual(traj_arg.observation.numpy(), initial_obs.reshape(1,2))
        self.assertTrue(traj_arg.is_mid()) # Because current_ts_val was a transition


if __name__ == '__main__':
    tf.test.main()
