"""Tests for BuildingDatasetPartition class."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import pytest

from smart_control.dataset.conftest import SKIP_REASON
from smart_control.dataset.conftest import TEST_DATASET
from smart_control.dataset.dataset_partition import BuildingDatasetPartition

#
# HIGH LEVEL TESTS FOR ALL PARTITIONS IN BUILDING "SB1"...
#

# pylint:disable=line-too-long
# fmt:off
PARTITION_PARAMETERS = [
  dict(
    partition_id='2022_a',
    actions_shape=(51852, 3),
    actions_range=('2022-01-01 00:00:00+00:00', '2022-06-30 00:55:00+00:00'),
    observations_shape=(51852, 1198),
    observations_range=('2022-01-01 00:00:00+00:00', '2022-06-30 00:55:00+00:00'),
    rewards_shape=(51852, 17),
    rewards_range=('2021-12-31 23:55:00+00:00', '2022-06-30 00:50:00+00:00'),
    reward_infos_shape=(51852, 3252),
    reward_infos_range=('2021-12-31 23:55:00+00:00', '2022-06-30 00:50:00+00:00'),
  ),
  dict(
    partition_id='2022_b',
    actions_shape=(53292, 3),
    actions_range=('2022-07-01 00:00:00+00:00', '2022-12-31 00:55:00+00:00'),
    observations_shape=(53292, 1198),
    observations_range=('2022-07-01 00:00:00+00:00', '2022-12-31 00:55:00+00:00'),
    rewards_shape=(53292, 17),
    rewards_range=('2022-06-30 23:55:00+00:00', '2022-12-31 00:50:00+00:00'),
    reward_infos_shape=(53292, 3318),
    reward_infos_range=('2022-06-30 23:55:00+00:00', '2022-12-31 00:50:00+00:00'),
  ),
  dict(
    partition_id='2023_a',
    actions_shape=(51852, 3),
    actions_range=('2023-01-01 00:00:00+00:00', '2023-06-30 00:55:00+00:00'),
    observations_shape=(51852, 1198),
    observations_range=('2023-01-01 00:00:00+00:00', '2023-06-30 00:55:00+00:00'),
    rewards_shape=(51852, 17),
    rewards_range=('2022-12-31 23:55:00+00:00', '2023-06-30 00:50:00+00:00'),
    reward_infos_shape=(51852, 3252),
    reward_infos_range=('2022-12-31 23:55:00+00:00', '2023-06-30 00:50:00+00:00'),
  ),
  dict(
    partition_id='2023_b',
    actions_shape=(52716, 3),
    actions_range=('2023-07-01 00:00:00+00:00', '2023-12-31 00:55:00+00:00'),
    observations_shape=(52716, 1198),
    observations_range=('2023-07-01 00:00:00+00:00', '2023-12-31 00:55:00+00:00'),
    rewards_shape=(52716, 17),
    rewards_range=('2023-06-30 23:55:00+00:00', '2023-12-31 00:50:00+00:00'),
    reward_infos_shape=(52716, 3252),
    reward_infos_range=('2023-06-30 23:55:00+00:00', '2023-12-31 00:50:00+00:00'),
  ),
  dict(
    partition_id='2024_a',
    actions_shape=(52140, 3),
    actions_range=('2024-01-01 00:00:00+00:00', '2024-06-30 00:55:00+00:00'),
    observations_shape=(52140, 1198),
    observations_range=('2024-01-01 00:00:00+00:00', '2024-06-30 00:55:00+00:00'),
    rewards_shape=(52140, 17),
    rewards_range=('2023-12-31 23:55:00+00:00', '2024-06-30 00:50:00+00:00'),
    reward_infos_shape=(52140, 3252),
    reward_infos_range=('2023-12-31 23:55:00+00:00', '2024-06-30 00:50:00+00:00'),
  ),
]
# pylint:enable=line-too-long
# fmt: on


@pytest.mark.usefixtures('set_dataset')
class TestAllBuildingDatasetPartitions(parameterized.TestCase):
  """Tests all valid partitions for building "sb1"."""

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  @parameterized.parameters(PARTITION_PARAMETERS)
  def test_all_partitions(
      self,
      partition_id,
      actions_shape,
      actions_range,
      observations_shape,
      observations_range,
      rewards_shape,
      rewards_range,
      reward_infos_shape,
      reward_infos_range,
  ):
    partition = BuildingDatasetPartition(self.ds, partition_id)

    actions_df = partition.actions_df
    self.assertIsInstance(actions_df, pd.DataFrame)
    self.assertEqual(actions_df.shape, actions_shape)
    self.assertEqual(str(actions_df.index[0]), actions_range[0])
    self.assertEqual(str(actions_df.index[-1]), actions_range[-1])

    observations_df = partition.observations_df
    self.assertIsInstance(observations_df, pd.DataFrame)
    self.assertEqual(observations_df.shape, observations_shape)
    self.assertEqual(str(observations_df.index[0]), observations_range[0])
    self.assertEqual(str(observations_df.index[-1]), observations_range[-1])

    rewards_df = partition.rewards_df
    self.assertIsInstance(rewards_df, pd.DataFrame)
    self.assertEqual(rewards_df.shape, rewards_shape)
    self.assertEqual(str(rewards_df.index[0]), rewards_range[0])
    self.assertEqual(str(rewards_df.index[-1]), rewards_range[-1])

    reward_infos_df = partition.reward_infos_df
    self.assertIsInstance(reward_infos_df, pd.DataFrame)
    self.assertEqual(reward_infos_df.shape, reward_infos_shape)
    self.assertEqual(str(reward_infos_df.index[0]), reward_infos_range[0])
    self.assertEqual(str(reward_infos_df.index[-1]), reward_infos_range[-1])


#
# DETAILED TESTS FOR THE "2022a" PARTITION...
#

_ACTION_IDS_MAP = {
    '12945159110931775488@supply_air_temperature_setpoint': 0,
    '13761436543392677888@supply_water_temperature_setpoint': 1,
    '14409954889734029312@supply_air_temperature_setpoint': 2,
}
_ACTION_IDS = list(_ACTION_IDS_MAP.keys())

_REWARD_IDS = [
    'rooms/9028552126@heating_setpoint_temperature',
    'rooms/9028552126@cooling_setpoint_temperature',
    'rooms/9028552126@zone_air_temperature',
    'rooms/9028552126@air_flow_rate_setpoint',
    'rooms/9028552126@air_flow_rate',
    'rooms/9028552126@average_occupancy',
    'rooms/9028472496@heating_setpoint_temperature',
    'rooms/9028472496@cooling_setpoint_temperature',
    'rooms/9028472496@zone_air_temperature',
    'rooms/9028472496@air_flow_rate_setpoint',
    'rooms/9028472496@air_flow_rate',
    'rooms/9028472496@average_occupancy',
    'rooms/9028552250@heating_setpoint_temperature',
    'rooms/9028552250@cooling_setpoint_temperature',
    'rooms/9028552250@zone_air_temperature',
    'rooms/9028552250@air_flow_rate_setpoint',
    'rooms/9028552250@air_flow_rate',
]


@pytest.mark.usefixtures('set_dataset')
@pytest.mark.usefixtures('set_partition')
class TestBuildingDatasetPartition(absltest.TestCase):
  """Tests for the BuildingDatasetPartition class."""

  def test_partition_validations(self):
    with self.assertRaises(ValueError):
      invalid_id = 'OOPS'
      BuildingDatasetPartition(
          building_dataset=self.ds, partition_id=invalid_id
      )

  def _assert_timestamps(self, timestamps, earliest, latest, length):
    """
    Assertions for timestamps.

    Args:
      timestamps (list): the timestamps to test
      earliest and latest (str): expected earliest and latest values,
        as strings, like '2022-06-30 00:55:00+00:00'
      length (int) : expected length of the list
    """
    self.assertIsInstance(timestamps, list)
    self.assertEqual(len(timestamps), length)

    first_timestamp = timestamps[0]
    last_timestamp = timestamps[-1]
    self.assertIsInstance(first_timestamp, pd.Timestamp)
    self.assertIsInstance(last_timestamp, pd.Timestamp)
    self.assertEqual(str(first_timestamp), earliest)
    self.assertEqual(str(last_timestamp), latest)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_partition_data(self):
    data = self.partition.data
    self.assertIsInstance(data, np.lib.npyio.NpzFile)

    # we are surfacing each key into its own high-level public property:
    with self.subTest('action_value_matrix'):
      np.testing.assert_array_equal(
          data['action_value_matrix'], self.partition.action_value_matrix
      )
    with self.subTest('observation_value_matrix'):
      np.testing.assert_array_equal(
          data['observation_value_matrix'],
          self.partition.observation_value_matrix,
      )
    with self.subTest('reward_value_matrix'):
      np.testing.assert_array_equal(
          data['reward_value_matrix'], self.partition.reward_value_matrix
      )
    with self.subTest('reward_info_value_matrix'):
      np.testing.assert_array_equal(
          data['reward_info_value_matrix'],
          self.partition.reward_info_value_matrix,
      )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_partition_metadata(self):
    metadata = self.partition.metadata

    self.assertIsInstance(metadata, dict)
    self.assertEqual(
        sorted(metadata.keys()),
        [
            'action_ids_map',
            'action_timestamps',
            'observation_ids_map',
            'observation_timestamps',
            'reward_ids_map',
            'reward_info_timestamps',
            'reward_timestamps',
        ],
    )

    # we are surfacing each key into its own high-level public property
    # fmt: off
    # pylint: disable=line-too-long
    self.assertEqual(metadata['action_ids_map'], self.partition.action_ids_map)
    self.assertEqual(metadata['observation_ids_map'], self.partition.observation_ids_map)
    self.assertEqual(metadata['reward_ids_map'], self.partition.reward_ids_map)

    self.assertEqual(metadata['action_timestamps'], self.partition.action_timestamps)
    self.assertEqual(metadata['observation_timestamps'], self.partition.observation_timestamps)
    self.assertEqual(metadata['reward_timestamps'], self.partition.reward_timestamps)
    self.assertEqual(metadata['reward_info_timestamps'], self.partition.reward_info_timestamps)
    # pylint: enable=line-too-long
    # fmt: on

  #
  # DATA PROPERTIES...
  #

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_action_value_matrix(self):
    self.assertEqual(self.partition.action_value_matrix.shape, (51852, 3))

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observation_value_matrix(self):
    self.assertEqual(self.partition.observation_value_matrix.shape, (51852, 1198))  # pylint: disable=line-too-long

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_reward_value_matrix(self):
    self.assertEqual(self.partition.reward_value_matrix.shape, (51852, 17))

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_reward_info_value_matrix(self):
    self.assertEqual(self.partition.reward_info_value_matrix.shape, (51852, 3252))  # pylint: disable=line-too-long

  #
  # METADATA PROPERTIES...
  #

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_action_ids_map(self):
    self.assertEqual(self.partition.action_ids_map, _ACTION_IDS_MAP)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observation_ids_map(self):
    observation_ids_map = self.partition.observation_ids_map
    self.assertIsInstance(observation_ids_map, dict)
    self.assertEqual(len(observation_ids_map), 1198)

    # keys are the observation ids:
    keys = list(observation_ids_map.keys())
    self.assertEqual(keys[0], '202194278473007104@building_air_static_pressure_setpoint')  # pylint: disable=line-too-long
    self.assertEqual(keys[-1], '2640423556868160@zone_air_temperature_sensor')

    # values are unique integers:
    values = list(observation_ids_map.values())
    self.assertEqual(values[0], 0)
    self.assertEqual(values[-1], 1197)
    self.assertEqual(len(values), len(list(set(values))))  # all unique

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_reward_ids_map(self):
    reward_ids_map = self.partition.reward_ids_map
    self.assertIsInstance(reward_ids_map, dict)
    self.assertEqual(len(reward_ids_map), 3252)

    # keys are the reward ids:
    keys = list(reward_ids_map.keys())
    self.assertEqual(keys[0], 'rooms/9028552126@heating_setpoint_temperature')
    self.assertEqual(keys[-1], '14409954889734029312@air_conditioning_electrical_energy_rate')  # pylint: disable=line-too-long

    # values are unique integers:
    values = list(reward_ids_map.values())
    self.assertEqual(values[0], 0)
    self.assertEqual(values[-1], 3251)
    self.assertEqual(len(values), len(list(set(values))))

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_action_ids(self):
    self.assertEqual(self.partition.action_ids, _ACTION_IDS)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observation_ids(self):
    self.assertEqual(
        self.partition.observation_ids,
        list(self.partition.observation_ids_map.keys()),
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_reward_ids(self):
    self.assertEqual(
        self.partition.reward_ids, list(self.partition.reward_ids_map.keys())
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_action_timestamps(self):
    self._assert_timestamps(
        self.partition.action_timestamps,
        earliest='2022-01-01 00:00:00+00:00',
        latest='2022-06-30 00:55:00+00:00',
        length=51852,
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observation_timestamps(self):
    self._assert_timestamps(
        self.partition.observation_timestamps,
        earliest='2022-01-01 00:00:00+00:00',
        latest='2022-06-30 00:55:00+00:00',
        length=51852,
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_reward_timestamps(self):
    self._assert_timestamps(
        self.partition.reward_timestamps,
        earliest='2021-12-31 23:55:00+00:00',
        latest='2022-06-30 00:50:00+00:00',
        length=51852,
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_reward_info_timestamps(self):
    self._assert_timestamps(
        self.partition.reward_info_timestamps,
        earliest='2021-12-31 23:55:00+00:00',
        latest='2022-06-30 00:50:00+00:00',
        length=51852,
    )

  #
  # DATAFRAME PROPERTIES...
  #

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observations_df(self):
    df = self.partition.observations_df

    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(df.shape, (51852, 1198))

    # columns corresponding to the observation ids:
    # ... there are 1198, but here are some examples:
    self.assertIn(
        '202194278473007104@building_air_static_pressure_setpoint', df.columns
    )
    self.assertIn('2640423556868160@zone_air_temperature_sensor', df.columns)

    # index corresponding to the observation timestamps:
    self.assertEqual(str(df.index[0]), '2022-01-01 00:00:00+00:00')
    self.assertEqual(str(df.index[-1]), '2022-06-30 00:55:00+00:00')

    # index is sorted in ascending order:
    self.assertEqual(df.index[0], df.index.min())
    self.assertEqual(df.index[-1], df.index.max())

    # values are numeric (float) and non-null:
    self.assertEqual(df.isna().sum().sum(), 0)
    self.assertEqual(df.dtypes.unique().tolist(), [np.dtype('float64')])

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_actions_df(self):
    df = self.partition.actions_df

    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(df.shape, (51852, 3))

    # columns corresponding to the action ids:
    self.assertEqual(df.columns.tolist(), _ACTION_IDS)

    # index corresponding to the action timestamps:
    self.assertEqual(str(df.index[0]), '2022-01-01 00:00:00+00:00')
    self.assertEqual(str(df.index[-1]), '2022-06-30 00:55:00+00:00')

    # index is sorted in ascending order:
    self.assertEqual(df.index[0], df.index.min())
    self.assertEqual(df.index[-1], df.index.max())

    # values are numeric (float) and non-null:
    self.assertEqual(df.isna().sum().sum(), 0)
    self.assertEqual(df.dtypes.unique().tolist(), [np.dtype('float64')])

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_rewards_df(self):
    df = self.partition.rewards_df

    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(df.shape, (51852, 17))

    # columns corresponding to the reward ids:
    self.assertEqual(df.columns.tolist(), _REWARD_IDS)

    # index corresponding to the reward timestamps:
    self.assertEqual(str(df.index[0]), '2021-12-31 23:55:00+00:00')
    self.assertEqual(str(df.index[-1]), '2022-06-30 00:50:00+00:00')

    # index is sorted in ascending order:
    self.assertEqual(df.index[0], df.index.min())
    self.assertEqual(df.index[-1], df.index.max())

    # values are numeric (float) and non-null:
    self.assertEqual(df.isna().sum().sum(), 0)  # all non-null
    self.assertEqual(df.dtypes.unique().tolist(), [np.dtype('float64')])

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_reward_infos_df(self):
    df = self.partition.reward_infos_df

    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(df.shape, (51852, 3252))

    # columns corresponding to the reward ids:
    # ... there are 3252 but here are some examples:
    self.assertIn('rooms/9028552126@heating_setpoint_temperature', df.columns)
    self.assertIn('14409954889734029312@air_conditioning_electrical_energy_rate', df.columns)  # pytest: disable=line-too-long # fmt:skip

    # index corresponding to the reward info timestamps:
    self.assertEqual(str(df.index[0]), '2021-12-31 23:55:00+00:00')
    self.assertEqual(str(df.index[-1]), '2022-06-30 00:50:00+00:00')

    # index is sorted in ascending order:
    self.assertEqual(df.index[0], df.index.min())
    self.assertEqual(df.index[-1], df.index.max())

    # values are numeric (float) and non-null:
    self.assertEqual(df.isna().sum().sum(), 0)
    self.assertEqual(df.dtypes.unique().tolist(), [np.dtype('float64')])


if __name__ == '__main__':
  absltest.main()
