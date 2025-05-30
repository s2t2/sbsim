import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import tensorflow as tf

from smart_control.reinforcement_learning.observers import rendering_observer
# BuildingRenderer will be mocked, so direct import not strictly needed if patching by string path
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step


class RenderingObserverTest(tf.test.TestCase):

    def _create_dummy_trajectory(self, observation_dict, action_val, step_type_val=time_step.StepType.MID, batched=False, batch_size=1):
        if batched:
            # Create batched versions of all trajectory components
            step_type = np.array([step_type_val] * batch_size, dtype=np.int32)
            # For observations, we need to stack each key's value in the dict
            observation = {k: np.array([v] * batch_size, dtype=np.float32) for k, v in observation_dict.items()}
            action = np.array([action_val] * batch_size, dtype=np.float32)
            policy_info = () # Keep simple for now
            next_step_type = np.array([time_step.StepType.MID] * batch_size, dtype=np.int32) # Arbitrary
            reward = np.array([0.0] * batch_size, dtype=np.float32) # Arbitrary
            discount = np.array([1.0] * batch_size, dtype=np.float32) # Arbitrary
        else:
            step_type = np.array(step_type_val, dtype=np.int32)
            observation = {k: np.array(v, dtype=np.float32) for k,v in observation_dict.items()}
            action = np.array(action_val, dtype=np.float32)
            policy_info = ()
            next_step_type = np.array(time_step.StepType.MID, dtype=np.int32)
            reward = np.array(0.0, dtype=np.float32)
            discount = np.array(1.0, dtype=np.float32)

        return trajectory.Trajectory(
            step_type=step_type,
            observation=observation,
            action=action,
            policy_info=policy_info,
            next_step_type=next_step_type,
            reward=reward,
            discount=discount
        )

    @patch('smart_control.reinforcement_learning.observers.rendering_observer.BuildingRenderer', autospec=True)
    def test_init(self, MockBuildingRenderer):
        mock_config = MagicMock(name="ConfigMock")
        priority = 10

        observer = rendering_observer.RenderingObserver(config=mock_config, priority=priority)

        # Verify _renderer is an instance of the mocked BuildingRenderer
        MockBuildingRenderer.assert_called_once_with(mock_config)
        self.assertIsInstance(observer._renderer, MockBuildingRenderer.return_value.__class__) # Check instance type

        # Verify priority
        self.assertEqual(observer.priority, priority)

    @patch('smart_control.reinforcement_learning.observers.rendering_observer.BuildingRenderer', autospec=True)
    def test_init_default_priority(self, MockBuildingRenderer):
        mock_config = MagicMock(name="ConfigMockDefaultPrio")
        observer = rendering_observer.RenderingObserver(config=mock_config)
        MockBuildingRenderer.assert_called_once_with(mock_config)
        self.assertEqual(observer.priority, 0) # Assuming default priority is 0

    @patch('smart_control.reinforcement_learning.observers.rendering_observer.BuildingRenderer', autospec=True)
    def test_call_single_trajectory_not_first(self, MockBuildingRenderer):
        mock_config = MagicMock(name="ConfigCall")
        observer = rendering_observer.RenderingObserver(config=mock_config)
        mock_renderer_instance = MockBuildingRenderer.return_value

        # Create a dummy trajectory - not a FIRST step
        obs_dict = {'sensor_1': [10.0], 'sensor_2': [20.0]}
        action_val = [0.5]
        dummy_traj = self._create_dummy_trajectory(
            observation_dict=obs_dict,
            action_val=action_val,
            step_type_val=time_step.StepType.MID # Not FIRST
        )

        observer(dummy_traj)

        # Verify renderer.render was called with observation and action
        mock_renderer_instance.render.assert_called_once()
        call_args, call_kwargs = mock_renderer_instance.render.call_args
        
        # The observer passes the observation and action from the trajectory
        # For a single, non-batched trajectory, these are directly the values.
        passed_observation = call_args[0]
        passed_action = call_args[1]

        # Check the observation dictionary
        self.assertIn('sensor_1', passed_observation)
        self.assertAllClose(passed_observation['sensor_1'], np.array(obs_dict['sensor_1'], dtype=np.float32))
        self.assertIn('sensor_2', passed_observation)
        self.assertAllClose(passed_observation['sensor_2'], np.array(obs_dict['sensor_2'], dtype=np.float32))
        
        self.assertAllClose(passed_action, np.array(action_val, dtype=np.float32))
        
        # Ensure reset was not called
        mock_renderer_instance.reset.assert_not_called()

    @patch('smart_control.reinforcement_learning.observers.rendering_observer.BuildingRenderer', autospec=True)
    def test_call_single_trajectory_is_first(self, MockBuildingRenderer):
        mock_config = MagicMock(name="ConfigCallFirst")
        observer = rendering_observer.RenderingObserver(config=mock_config)
        mock_renderer_instance = MockBuildingRenderer.return_value

        obs_dict = {'sensor_1': [5.0], 'sensor_2': [15.0]}
        action_val = [0.1]
        dummy_traj = self._create_dummy_trajectory(
            observation_dict=obs_dict,
            action_val=action_val,
            step_type_val=time_step.StepType.FIRST # IS FIRST step
        )

        observer(dummy_traj)

        # Verify renderer.reset was called
        mock_renderer_instance.reset.assert_called_once()

        # Verify renderer.render was also called (as per current implementation)
        mock_renderer_instance.render.assert_called_once()
        call_args, _ = mock_renderer_instance.render.call_args
        passed_observation = call_args[0]
        passed_action = call_args[1]
        self.assertAllClose(passed_observation['sensor_1'], np.array(obs_dict['sensor_1'], dtype=np.float32))
        self.assertAllClose(passed_action, np.array(action_val, dtype=np.float32))

    @patch('smart_control.reinforcement_learning.observers.rendering_observer.BuildingRenderer', autospec=True)
    def test_call_batched_trajectory_renders_first_element(self, MockBuildingRenderer):
        mock_config = MagicMock(name="ConfigCallBatched")
        observer = rendering_observer.RenderingObserver(config=mock_config)
        mock_renderer_instance = MockBuildingRenderer.return_value

        # Create two distinct observations and actions for the batch
        obs_dict1 = {'sensor_1': [1.0], 'sensor_2': [2.0]}
        action_val1 = [0.1]
        obs_dict2 = {'sensor_1': [3.0], 'sensor_2': [4.0]} # Different values for second item
        action_val2 = [0.2]

        # Create a batched trajectory with batch_size=2
        # First item is MID, second is also MID (or any other non-FIRST)
        batched_traj = self._create_dummy_trajectory(
            observation_dict=obs_dict1, # This will be used for all items due to current helper
            action_val=action_val1,     # This will be used for all items
            step_type_val=time_step.StepType.MID,
            batched=True,
            batch_size=2
        )
        # Manually update the second item's observation and action in the batched trajectory
        # to ensure they are different for the test.
        # _create_dummy_trajectory currently makes all items in batch identical.
        current_obs = batched_traj.observation
        current_act = batched_traj.action
        current_obs['sensor_1'][1] = np.array(obs_dict2['sensor_1'], dtype=np.float32)
        current_obs['sensor_2'][1] = np.array(obs_dict2['sensor_2'], dtype=np.float32)
        current_act[1] = np.array(action_val2, dtype=np.float32)
        
        final_batched_traj = batched_traj._replace(observation=current_obs, action=current_act)

        observer(final_batched_traj)

        # Verify render was called once (as it only processes the first item)
        mock_renderer_instance.render.assert_called_once()
        call_args, _ = mock_renderer_instance.render.call_args
        passed_observation = call_args[0]
        passed_action = call_args[1]

        # Check that the passed observation and action are from the FIRST item in the batch
        self.assertAllClose(passed_observation['sensor_1'], np.array(obs_dict1['sensor_1'], dtype=np.float32))
        self.assertAllClose(passed_observation['sensor_2'], np.array(obs_dict1['sensor_2'], dtype=np.float32))
        self.assertAllClose(passed_action, np.array(action_val1, dtype=np.float32))
        
        mock_renderer_instance.reset.assert_not_called() # Since first item is MID

    @patch('smart_control.reinforcement_learning.observers.rendering_observer.BuildingRenderer', autospec=True)
    def test_call_batched_trajectory_first_item_is_first_step(self, MockBuildingRenderer):
        mock_config = MagicMock(name="ConfigCallBatchedFirst")
        observer = rendering_observer.RenderingObserver(config=mock_config)
        mock_renderer_instance = MockBuildingRenderer.return_value

        obs_dict1 = {'sensor_1': [10.0], 'sensor_2': [20.0]}
        action_val1 = [0.5]
        
        batched_traj = self._create_dummy_trajectory(
            observation_dict=obs_dict1,
            action_val=action_val1,
            step_type_val=time_step.StepType.FIRST, # First item in batch is FIRST
            batched=True,
            batch_size=2
        )
        # If second item in batch was different and FIRST, it shouldn't matter for reset
        # as reset is based on the overall trajectory.is_first() which checks the first element of batch.

        observer(batched_traj)

        mock_renderer_instance.reset.assert_called_once()
        mock_renderer_instance.render.assert_called_once() # Render is still called for the first item
        call_args, _ = mock_renderer_instance.render.call_args
        passed_observation = call_args[0]
        passed_action = call_args[1]
        self.assertAllClose(passed_observation['sensor_1'], np.array(obs_dict1['sensor_1'], dtype=np.float32))
        self.assertAllClose(passed_action, np.array(action_val1, dtype=np.float32))


if __name__ == '__main__':
    tf.test.main()
