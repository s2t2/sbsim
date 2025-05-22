import unittest
from unittest.mock import patch, MagicMock

import tensorflow as tf
import pandas as pd
import numpy as np

from smart_control.dataset import dataset
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory


class DatasetTest(tf.test.TestCase):
    @patch('pandas.read_csv')
    def test_init_single_file_discrete_action(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action': [0, 1, 0]})
        mock_observation_df = pd.DataFrame({'obs1': [0.1, 0.2, 0.3], 'obs2': [0.4, 0.5, 0.6]})
        # side_effect function to return different dataframes based on file path
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path' # Does not matter due to mocking
        batch_size = 1
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        # Initialize Dataset
        ds = dataset.Dataset(
            file_path=file_path,
            batch_size=batch_size,
            is_single_file=is_single_file,
            is_action_discrete=is_action_discrete,
            num_shards=num_shards,
            action_name=action_name,
            observation_names=observation_names
        )

        # Assertions
        self.assertEqual(ds._length, 3) # Length of the mock dataframes
        self.assertEqual(ds._file_paths_action, [f'{file_path}/action.csv'])
        self.assertEqual(ds._file_paths_observation, [f'{file_path}/observation.csv'])
        self.assertTrue(ds._is_action_discrete)

        # Check specs
        self.assertIsInstance(ds._action_spec, array_spec.BoundedArraySpec)
        self.assertEqual(ds._action_spec.shape, ())
        self.assertEqual(ds._action_spec.dtype, np.int64)
        self.assertEqual(ds._action_spec.minimum, 0)
        self.assertEqual(ds._action_spec.maximum, 1) # Max unique value in mock_action_df['action']

        self.assertIsInstance(ds._observation_spec, dict)
        self.assertLen(ds._observation_spec, 2)
        self.assertIsInstance(ds._observation_spec['obs1'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs1'].shape, ())
        self.assertEqual(ds._observation_spec['obs1'].dtype, np.float32) # from dataframe
        self.assertIsInstance(ds._observation_spec['obs2'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs2'].shape, ())
        self.assertEqual(ds._observation_spec['obs2'].dtype, np.float32) # from dataframe

        self.assertIsInstance(ds.data_spec(), trajectory.Trajectory)

    @patch('pandas.read_csv')
    def test_init_single_file_continuous_action(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action1': [0.1, 0.2, 0.3], 'action2': [0.4, 0.5, 0.6]})
        mock_observation_df = pd.DataFrame({'obs1': [0.7, 0.8, 0.9], 'obs2': [1.0, 1.1, 1.2]})
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path'
        batch_size = 1
        is_single_file = True
        is_action_discrete = False # Continuous action
        num_shards = 1
        action_name = ['action1', 'action2'] # List for continuous actions
        observation_names = ['obs1', 'obs2']

        # Initialize Dataset
        ds = dataset.Dataset(
            file_path=file_path,
            batch_size=batch_size,
            is_single_file=is_single_file,
            is_action_discrete=is_action_discrete,
            num_shards=num_shards,
            action_name=action_name,
            observation_names=observation_names
        )

        # Assertions
        self.assertEqual(ds._length, 3)
        self.assertEqual(ds._file_paths_action, [f'{file_path}/action.csv'])
        self.assertEqual(ds._file_paths_observation, [f'{file_path}/observation.csv'])
        self.assertFalse(ds._is_action_discrete)

        # Check specs
        self.assertIsInstance(ds._action_spec, array_spec.ArraySpec) # Not BoundedArraySpec for continuous
        self.assertEqual(ds._action_spec.shape, (2,)) # Shape reflects number of action columns
        self.assertEqual(ds._action_spec.dtype, np.float32)

        self.assertIsInstance(ds._observation_spec, dict)
        self.assertLen(ds._observation_spec, 2)
        self.assertIsInstance(ds._observation_spec['obs1'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs1'].shape, ())
        self.assertEqual(ds._observation_spec['obs1'].dtype, np.float32)
        self.assertIsInstance(ds._observation_spec['obs2'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs2'].shape, ())
        self.assertEqual(ds._observation_spec['obs2'].dtype, np.float32)

        self.assertIsInstance(ds.data_spec(), trajectory.Trajectory)

    @patch('pandas.read_csv')
    def test_init_multiple_files_discrete_action(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df_shard0 = pd.DataFrame({'action': [0, 1]})
        mock_action_df_shard1 = pd.DataFrame({'action': [0]})
        mock_observation_df_shard0 = pd.DataFrame({'obs1': [0.1, 0.2], 'obs2': [0.3, 0.4]})
        mock_observation_df_shard1 = pd.DataFrame({'obs1': [0.5], 'obs2': [0.6]})

        def side_effect_read_csv(file_path, index_col=None):
            if 'action_0.csv' in file_path:
                return mock_action_df_shard0
            elif 'action_1.csv' in file_path:
                return mock_action_df_shard1
            elif 'observation_0.csv' in file_path:
                return mock_observation_df_shard0
            elif 'observation_1.csv' in file_path:
                return mock_observation_df_shard1
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path_sharded'
        batch_size = 1
        is_single_file = False # Multiple files
        is_action_discrete = True
        num_shards = 2
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        # Initialize Dataset
        ds = dataset.Dataset(
            file_path=file_path,
            batch_size=batch_size,
            is_single_file=is_single_file,
            is_action_discrete=is_action_discrete,
            num_shards=num_shards,
            action_name=action_name,
            observation_names=observation_names
        )

        # Assertions
        self.assertEqual(ds._length, 3) # 2 from shard0 + 1 from shard1
        expected_action_paths = [f'{file_path}/action_{i}.csv' for i in range(num_shards)]
        expected_observation_paths = [f'{file_path}/observation_{i}.csv' for i in range(num_shards)]
        self.assertEqual(ds._file_paths_action, expected_action_paths)
        self.assertEqual(ds._file_paths_observation, expected_observation_paths)
        self.assertTrue(ds._is_action_discrete)

        # Check specs (based on combined data from shards)
        self.assertIsInstance(ds._action_spec, array_spec.BoundedArraySpec)
        self.assertEqual(ds._action_spec.shape, ())
        self.assertEqual(ds._action_spec.dtype, np.int64)
        self.assertEqual(ds._action_spec.minimum, 0) # min(0,1,0)
        self.assertEqual(ds._action_spec.maximum, 1) # max(0,1,0)

        self.assertIsInstance(ds._observation_spec, dict)
        self.assertLen(ds._observation_spec, 2)
        self.assertIsInstance(ds._observation_spec['obs1'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs1'].shape, ())
        self.assertEqual(ds._observation_spec['obs1'].dtype, np.float32)
        self.assertIsInstance(ds._observation_spec['obs2'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs2'].shape, ())
        self.assertEqual(ds._observation_spec['obs2'].dtype, np.float32)

        self.assertIsInstance(ds.data_spec(), trajectory.Trajectory)

    @patch('pandas.read_csv')
    def test_init_multiple_files_continuous_action(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df_shard0 = pd.DataFrame({'action1': [0.1, 0.2], 'action2': [0.3, 0.4]})
        mock_action_df_shard1 = pd.DataFrame({'action1': [0.5], 'action2': [0.6]})
        mock_observation_df_shard0 = pd.DataFrame({'obs1': [0.7, 0.8], 'obs2': [0.9, 1.0]})
        mock_observation_df_shard1 = pd.DataFrame({'obs1': [1.1], 'obs2': [1.2]})

        def side_effect_read_csv(file_path, index_col=None):
            if 'action_0.csv' in file_path:
                return mock_action_df_shard0
            elif 'action_1.csv' in file_path:
                return mock_action_df_shard1
            elif 'observation_0.csv' in file_path:
                return mock_observation_df_shard0
            elif 'observation_1.csv' in file_path:
                return mock_observation_df_shard1
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path_sharded'
        batch_size = 1
        is_single_file = False # Multiple files
        is_action_discrete = False # Continuous action
        num_shards = 2
        action_name = ['action1', 'action2'] # List for continuous actions
        observation_names = ['obs1', 'obs2']

        # Initialize Dataset
        ds = dataset.Dataset(
            file_path=file_path,
            batch_size=batch_size,
            is_single_file=is_single_file,
            is_action_discrete=is_action_discrete,
            num_shards=num_shards,
            action_name=action_name,
            observation_names=observation_names
        )

        # Assertions
        self.assertEqual(ds._length, 3) # 2 from shard0 + 1 from shard1
        expected_action_paths = [f'{file_path}/action_{i}.csv' for i in range(num_shards)]
        expected_observation_paths = [f'{file_path}/observation_{i}.csv' for i in range(num_shards)]
        self.assertEqual(ds._file_paths_action, expected_action_paths)
        self.assertEqual(ds._file_paths_observation, expected_observation_paths)
        self.assertFalse(ds._is_action_discrete)

        # Check specs (based on combined data from shards)
        self.assertIsInstance(ds._action_spec, array_spec.ArraySpec)
        self.assertEqual(ds._action_spec.shape, (2,)) # Two action columns
        self.assertEqual(ds._action_spec.dtype, np.float32)

        self.assertIsInstance(ds._observation_spec, dict)
        self.assertLen(ds._observation_spec, 2)
        self.assertIsInstance(ds._observation_spec['obs1'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs1'].shape, ())
        self.assertEqual(ds._observation_spec['obs1'].dtype, np.float32)
        self.assertIsInstance(ds._observation_spec['obs2'], array_spec.ArraySpec)
        self.assertEqual(ds._observation_spec['obs2'].shape, ())
        self.assertEqual(ds._observation_spec['obs2'].dtype, np.float32)

        self.assertIsInstance(ds.data_spec(), trajectory.Trajectory)

    @patch('pandas.read_csv')
    def test_next_single_file_discrete_action_batch_1(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action': [0, 1, 0]})
        mock_observation_df = pd.DataFrame({'obs1': [0.1, 0.2, 0.3], 'obs2': [0.4, 0.5, 0.6]})
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path'
        batch_size = 1
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        # Iterate and check trajectories
        expected_observations = [
            {'obs1': np.array([0.1], dtype=np.float32), 'obs2': np.array([0.4], dtype=np.float32)},
            {'obs1': np.array([0.2], dtype=np.float32), 'obs2': np.array([0.5], dtype=np.float32)},
            {'obs1': np.array([0.3], dtype=np.float32), 'obs2': np.array([0.6], dtype=np.float32)},
        ]
        expected_actions = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ]

        for i in range(ds._length):
            traj = next(ds)
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertEqual(traj.step_type.numpy().item(), 0) # FIRST
            self.assertAllClose(traj.observation['obs1'].numpy(), expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), expected_observations[i]['obs2'])
            self.assertAllEqual(traj.action.numpy(), expected_actions[i])
            self.assertEqual(traj.next_step_type.numpy().item(), 1) # MID or LAST
            self.assertAllClose(traj.reward.numpy(), np.array([0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0], dtype=np.float32))

        # Test StopIteration
        with self.assertRaises(StopIteration):
            next(ds)

    @patch('pandas.read_csv')
    def test_next_single_file_continuous_action_batch_1(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action1': [0.1, 0.3], 'action2': [0.2, 0.4]})
        mock_observation_df = pd.DataFrame({'obs1': [0.5, 0.7], 'obs2': [0.6, 0.8]})
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path'
        batch_size = 1
        is_single_file = True
        is_action_discrete = False
        num_shards = 1
        action_name = ['action1', 'action2']
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        # Iterate and check trajectories
        expected_observations = [
            {'obs1': np.array([0.5], dtype=np.float32), 'obs2': np.array([0.6], dtype=np.float32)},
            {'obs1': np.array([0.7], dtype=np.float32), 'obs2': np.array([0.8], dtype=np.float32)},
        ]
        expected_actions = [
            np.array([[0.1, 0.2]], dtype=np.float32), # Note the extra dimension for batch
            np.array([[0.3, 0.4]], dtype=np.float32),
        ]

        for i in range(ds._length):
            traj = next(ds)
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertEqual(traj.step_type.numpy().item(), 0) # FIRST
            self.assertAllClose(traj.observation['obs1'].numpy(), expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), expected_observations[i]['obs2'])
            self.assertAllClose(traj.action.numpy(), expected_actions[i])
            self.assertEqual(traj.next_step_type.numpy().item(), 1) # MID or LAST
            self.assertAllClose(traj.reward.numpy(), np.array([0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0], dtype=np.float32))

        # Test StopIteration
        with self.assertRaises(StopIteration):
            next(ds)

    @patch('pandas.read_csv')
    def test_next_multiple_files_discrete_action_batch_1(self, mock_read_csv):
        # Mocking pd.read_csv for sharded data
        mock_action_df_shard0 = pd.DataFrame({'action': [0, 1]})
        mock_action_df_shard1 = pd.DataFrame({'action': [0]})
        mock_observation_df_shard0 = pd.DataFrame({'obs1': [0.1, 0.2], 'obs2': [0.3, 0.4]})
        mock_observation_df_shard1 = pd.DataFrame({'obs1': [0.5], 'obs2': [0.6]})

        def side_effect_read_csv(file_path, index_col=None):
            if 'action_0.csv' in file_path:
                return mock_action_df_shard0
            elif 'action_1.csv' in file_path:
                return mock_action_df_shard1
            elif 'observation_0.csv' in file_path:
                return mock_observation_df_shard0
            elif 'observation_1.csv' in file_path:
                return mock_observation_df_shard1
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path_sharded'
        batch_size = 1
        is_single_file = False
        is_action_discrete = True
        num_shards = 2
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        # Iterate and check trajectories
        # Data from shard 0
        expected_shard0_observations = [
            {'obs1': np.array([0.1], dtype=np.float32), 'obs2': np.array([0.3], dtype=np.float32)},
            {'obs1': np.array([0.2], dtype=np.float32), 'obs2': np.array([0.4], dtype=np.float32)},
        ]
        expected_shard0_actions = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
        ]
        # Data from shard 1
        expected_shard1_observations = [
            {'obs1': np.array([0.5], dtype=np.float32), 'obs2': np.array([0.6], dtype=np.float32)},
        ]
        expected_shard1_actions = [
            np.array([0], dtype=np.int64),
        ]

        all_expected_observations = expected_shard0_observations + expected_shard1_observations
        all_expected_actions = expected_shard0_actions + expected_shard1_actions

        for i in range(ds._length):
            traj = next(ds)
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertEqual(traj.step_type.numpy().item(), 0) # FIRST
            self.assertAllClose(traj.observation['obs1'].numpy(), all_expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), all_expected_observations[i]['obs2'])
            self.assertAllEqual(traj.action.numpy(), all_expected_actions[i])
            self.assertEqual(traj.next_step_type.numpy().item(), 1) # MID or LAST
            self.assertAllClose(traj.reward.numpy(), np.array([0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0], dtype=np.float32))

        # Test StopIteration
        with self.assertRaises(StopIteration):
            next(ds)

    @patch('pandas.read_csv')
    def test_next_multiple_files_continuous_action_batch_1(self, mock_read_csv):
        # Mocking pd.read_csv for sharded data
        mock_action_df_shard0 = pd.DataFrame({'action1': [0.1, 0.2], 'action2': [0.3, 0.4]})
        mock_action_df_shard1 = pd.DataFrame({'action1': [0.5], 'action2': [0.6]})
        mock_observation_df_shard0 = pd.DataFrame({'obs1': [0.7, 0.8], 'obs2': [0.9, 1.0]})
        mock_observation_df_shard1 = pd.DataFrame({'obs1': [1.1], 'obs2': [1.2]})

        def side_effect_read_csv(file_path, index_col=None):
            if 'action_0.csv' in file_path:
                return mock_action_df_shard0
            elif 'action_1.csv' in file_path:
                return mock_action_df_shard1
            elif 'observation_0.csv' in file_path:
                return mock_observation_df_shard0
            elif 'observation_1.csv' in file_path:
                return mock_observation_df_shard1
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path_sharded'
        batch_size = 1
        is_single_file = False
        is_action_discrete = False
        num_shards = 2
        action_name = ['action1', 'action2']
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        # Iterate and check trajectories
        # Data from shard 0
        expected_shard0_observations = [
            {'obs1': np.array([0.7], dtype=np.float32), 'obs2': np.array([0.9], dtype=np.float32)},
            {'obs1': np.array([0.8], dtype=np.float32), 'obs2': np.array([1.0], dtype=np.float32)},
        ]
        expected_shard0_actions = [
            np.array([[0.1, 0.3]], dtype=np.float32),
            np.array([[0.2, 0.4]], dtype=np.float32),
        ]
        # Data from shard 1
        expected_shard1_observations = [
            {'obs1': np.array([1.1], dtype=np.float32), 'obs2': np.array([1.2], dtype=np.float32)},
        ]
        expected_shard1_actions = [
            np.array([[0.5, 0.6]], dtype=np.float32),
        ]

        all_expected_observations = expected_shard0_observations + expected_shard1_observations
        all_expected_actions = expected_shard0_actions + expected_shard1_actions

        for i in range(ds._length):
            traj = next(ds)
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertEqual(traj.step_type.numpy().item(), 0) # FIRST
            self.assertAllClose(traj.observation['obs1'].numpy(), all_expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), all_expected_observations[i]['obs2'])
            self.assertAllClose(traj.action.numpy(), all_expected_actions[i])
            self.assertEqual(traj.next_step_type.numpy().item(), 1) # MID or LAST
            self.assertAllClose(traj.reward.numpy(), np.array([0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0], dtype=np.float32))

        # Test StopIteration
        with self.assertRaises(StopIteration):
            next(ds)

    @patch('pandas.read_csv')
    def test_next_single_file_discrete_action_batch_2(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action': [0, 1, 0, 1]}) # 4 samples for batch_size=2
        mock_observation_df = pd.DataFrame({
            'obs1': [0.1, 0.2, 0.3, 0.4],
            'obs2': [0.5, 0.6, 0.7, 0.8]
        })
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path'
        batch_size = 2
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        # Iterate and check trajectories
        # Batch 1
        expected_obs_batch1 = {
            'obs1': np.array([0.1, 0.2], dtype=np.float32),
            'obs2': np.array([0.5, 0.6], dtype=np.float32)
        }
        expected_action_batch1 = np.array([0, 1], dtype=np.int64)

        # Batch 2
        expected_obs_batch2 = {
            'obs1': np.array([0.3, 0.4], dtype=np.float32),
            'obs2': np.array([0.7, 0.8], dtype=np.float32)
        }
        expected_action_batch2 = np.array([0, 1], dtype=np.int64)

        all_expected_observations = [expected_obs_batch1, expected_obs_batch2]
        all_expected_actions = [expected_action_batch1, expected_action_batch2]

        for i in range(ds._length // batch_size): # Iterate for number of batches
            traj = next(ds)
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertAllEqual(traj.step_type.numpy(), [0, 0]) # FIRST for both items in batch
            self.assertAllClose(traj.observation['obs1'].numpy(), all_expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), all_expected_observations[i]['obs2'])
            self.assertAllEqual(traj.action.numpy(), all_expected_actions[i])
            self.assertAllEqual(traj.next_step_type.numpy(), [1, 1]) # MID or LAST
            self.assertAllClose(traj.reward.numpy(), np.array([0.0, 0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0, 1.0], dtype=np.float32))

        # Test StopIteration
        with self.assertRaises(StopIteration):
            next(ds)

    @patch('pandas.read_csv')
    def test_next_multiple_files_discrete_action_batch_2(self, mock_read_csv):
        # Mocking pd.read_csv for sharded data
        mock_action_df_shard0 = pd.DataFrame({'action': [0, 1]}) # 2 samples
        mock_action_df_shard1 = pd.DataFrame({'action': [0, 1]}) # 2 samples
        mock_observation_df_shard0 = pd.DataFrame({
            'obs1': [0.1, 0.2], 'obs2': [0.3, 0.4]
        })
        mock_observation_df_shard1 = pd.DataFrame({
            'obs1': [0.5, 0.6], 'obs2': [0.7, 0.8]
        })

        def side_effect_read_csv(file_path, index_col=None):
            if 'action_0.csv' in file_path:
                return mock_action_df_shard0
            elif 'action_1.csv' in file_path:
                return mock_action_df_shard1
            elif 'observation_0.csv' in file_path:
                return mock_observation_df_shard0
            elif 'observation_1.csv' in file_path:
                return mock_observation_df_shard1
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path_sharded'
        batch_size = 2
        is_single_file = False
        is_action_discrete = True
        num_shards = 2
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        # Iterate and check trajectories
        # Batch 1 (from shard 0)
        expected_obs_batch1 = {
            'obs1': np.array([0.1, 0.2], dtype=np.float32),
            'obs2': np.array([0.3, 0.4], dtype=np.float32)
        }
        expected_action_batch1 = np.array([0, 1], dtype=np.int64)

        # Batch 2 (from shard 1)
        expected_obs_batch2 = {
            'obs1': np.array([0.5, 0.6], dtype=np.float32),
            'obs2': np.array([0.7, 0.8], dtype=np.float32)
        }
        expected_action_batch2 = np.array([0, 1], dtype=np.int64)

        all_expected_observations = [expected_obs_batch1, expected_obs_batch2]
        all_expected_actions = [expected_action_batch1, expected_action_batch2]

        for i in range(ds._length // batch_size): # Iterate for number of batches
            traj = next(ds)
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertAllEqual(traj.step_type.numpy(), [0, 0])
            self.assertAllClose(traj.observation['obs1'].numpy(), all_expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), all_expected_observations[i]['obs2'])
            self.assertAllEqual(traj.action.numpy(), all_expected_actions[i])
            self.assertAllEqual(traj.next_step_type.numpy(), [1, 1])
            self.assertAllClose(traj.reward.numpy(), np.array([0.0, 0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0, 1.0], dtype=np.float32))

        # Test StopIteration
        with self.assertRaises(StopIteration):
            next(ds)

    @patch('pandas.read_csv')
    def test_get_tf_dataset_single_file_discrete_batch_1(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action': [0, 1, 0]})
        mock_observation_df = pd.DataFrame({'obs1': [0.1, 0.2, 0.3], 'obs2': [0.4, 0.5, 0.6]})
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path'
        batch_size = 1
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds_iterator = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        tf_ds = ds_iterator.get_tf_dataset()

        # Verify dataset type and element_spec
        self.assertIsInstance(tf_ds, tf.data.Dataset)
        self.assertEqual(tf_ds.element_spec, ds_iterator.data_spec())

        # Iterate a few elements and verify
        expected_observations = [
            {'obs1': np.array([0.1], dtype=np.float32), 'obs2': np.array([0.4], dtype=np.float32)},
            {'obs1': np.array([0.2], dtype=np.float32), 'obs2': np.array([0.5], dtype=np.float32)},
            {'obs1': np.array([0.3], dtype=np.float32), 'obs2': np.array([0.6], dtype=np.float32)},
        ]
        expected_actions = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ]

        for i, traj in enumerate(tf_ds.take(ds_iterator._length)):
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertEqual(traj.step_type.numpy().item(), 0) # FIRST
            self.assertAllClose(traj.observation['obs1'].numpy(), expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), expected_observations[i]['obs2'])
            self.assertAllEqual(traj.action.numpy(), expected_actions[i])
            self.assertEqual(traj.next_step_type.numpy().item(), 1) # MID or LAST
            self.assertAllClose(traj.reward.numpy(), np.array([0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0], dtype=np.float32))

    @patch('pandas.read_csv')
    def test_get_tf_dataset_single_file_discrete_batch_2(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action': [0, 1, 0, 1]}) # 4 samples for batch_size=2
        mock_observation_df = pd.DataFrame({
            'obs1': [0.1, 0.2, 0.3, 0.4],
            'obs2': [0.5, 0.6, 0.7, 0.8]
        })
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path'
        batch_size = 2
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds_iterator = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )
        tf_ds = ds_iterator.get_tf_dataset()

        self.assertIsInstance(tf_ds, tf.data.Dataset)
        self.assertEqual(tf_ds.element_spec, ds_iterator.data_spec())

        # Iterate and check trajectories
        # Batch 1
        expected_obs_batch1 = {
            'obs1': np.array([0.1, 0.2], dtype=np.float32),
            'obs2': np.array([0.5, 0.6], dtype=np.float32)
        }
        expected_action_batch1 = np.array([0, 1], dtype=np.int64)
        # Batch 2
        expected_obs_batch2 = {
            'obs1': np.array([0.3, 0.4], dtype=np.float32),
            'obs2': np.array([0.7, 0.8], dtype=np.float32)
        }
        expected_action_batch2 = np.array([0, 1], dtype=np.int64)

        all_expected_observations = [expected_obs_batch1, expected_obs_batch2]
        all_expected_actions = [expected_action_batch1, expected_action_batch2]

        for i, traj in enumerate(tf_ds.take(ds_iterator._length // batch_size)):
            self.assertIsInstance(traj, trajectory.Trajectory)
            self.assertAllEqual(traj.step_type.numpy(), [0, 0])
            self.assertAllClose(traj.observation['obs1'].numpy(), all_expected_observations[i]['obs1'])
            self.assertAllClose(traj.observation['obs2'].numpy(), all_expected_observations[i]['obs2'])
            self.assertAllEqual(traj.action.numpy(), all_expected_actions[i])
            self.assertAllEqual(traj.next_step_type.numpy(), [1, 1])
            self.assertAllClose(traj.reward.numpy(), np.array([0.0, 0.0], dtype=np.float32))
            self.assertAllClose(traj.discount.numpy(), np.array([1.0, 1.0], dtype=np.float32))

    @patch('pandas.read_csv')
    def test_get_num_steps(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action': [0, 1, 0]})
        mock_observation_df = pd.DataFrame({'obs1': [0.1, 0.2, 0.3], 'obs2': [0.4, 0.5, 0.6]})
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path'
        batch_size = 1
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )

        self.assertEqual(ds.get_num_steps(), 3) # Length of the mock dataframes

    @patch('pandas.read_csv')
    def test_init_empty_csv(self, mock_read_csv):
        # Mocking pd.read_csv to return empty dataframes
        mock_read_csv.return_value = pd.DataFrame()

        # Dataset parameters
        file_path = 'dummy_path_empty'
        batch_size = 1
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        with self.assertRaisesRegex(ValueError, "Dataframe is empty for action"):
            dataset.Dataset(
                file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
                is_action_discrete=is_action_discrete, num_shards=num_shards,
                action_name=action_name, observation_names=observation_names
            )

    @patch('pandas.read_csv')
    def test_init_single_row_csv(self, mock_read_csv):
        # Mocking pd.read_csv
        mock_action_df = pd.DataFrame({'action': [0]})
        mock_observation_df = pd.DataFrame({'obs1': [0.1], 'obs2': [0.4]})
        def side_effect_read_csv(file_path, index_col=None):
            if 'action' in file_path:
                return mock_action_df
            elif 'observation' in file_path:
                return mock_observation_df
            return pd.DataFrame()
        mock_read_csv.side_effect = side_effect_read_csv

        # Dataset parameters
        file_path = 'dummy_path_single_row'
        batch_size = 1
        is_single_file = True
        is_action_discrete = True
        num_shards = 1
        action_name = 'action'
        observation_names = ['obs1', 'obs2']

        ds = dataset.Dataset(
            file_path=file_path, batch_size=batch_size, is_single_file=is_single_file,
            is_action_discrete=is_action_discrete, num_shards=num_shards,
            action_name=action_name, observation_names=observation_names
        )
        self.assertEqual(ds._length, 1)
        traj = next(ds) # Check if one trajectory can be fetched
        self.assertIsInstance(traj, trajectory.Trajectory)
        with self.assertRaises(StopIteration): # Should stop after one
            next(ds)


if __name__ == '__main__':
    unittest.main()
