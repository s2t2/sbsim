"""Tests for the Smart Buildings Dataset.

Includes high fidelity tests to download the dataset and verify its structure.

It takes around two minutes to download and unzip the data, so we are skipping
dataset tests by default, to keep the build fast. But you can trigger a download
by setting the `TEST_DATASET_DOWNLOAD` environment variable to 'true'.

Downloaded data will not get cleared after tests run, so we can use it in
subsequent test runs without needing to re-download. This allows developers to
run dataset tests fairly quickly on their local machines. When the dataset
already exists locally, the tests take around five seconds.

The dataset tests will be run if the data is being downloaded, or if there is
existing local data.

Downloaded data will not get cleared by default before tests run, but you can
force a clean up and fresh download by setting the `CLEAR_TEST_DATASET_DOWNLOAD`
environment variable to 'true'.

These real tests are meant to be run periodically, for example once per day.
"""

import os
import shutil
import unittest

from absl.testing import absltest
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from smart_control.dataset.dataset import BuildingDataset
from smart_control.dataset.dataset import BuildingDatasetPartition
from smart_control.dataset.dataset import DATA_DIR

load_dotenv()

# whether or not to download the dataset:
TEST_DATASET_DOWNLOAD = bool(os.getenv('TEST_DATASET_DOWNLOAD', default='false').lower() == 'true')  # pylint: disable=line-too-long
# whether or not to delete existing local data before downloading:
CLEAR_TEST_DATASET_DOWNLOAD = bool(os.getenv('CLEAR_TEST_DATASET_DOWNLOAD', default='false').lower() == 'true')  # pylint: disable=line-too-long

DATASET_DIRPATH = os.path.join(DATA_DIR, 'sb1')
ZIP_FILEPATH = os.path.join(DATA_DIR, 'sb1.zip')

# whether or not to run dataset tests:
TEST_DATASET = bool(TEST_DATASET_DOWNLOAD or os.path.isdir(DATASET_DIRPATH))
SKIP_REASON = 'Skip large download by default.'


#
# SHARED FIXTURES & DATA STRUCTURES
#

_dataset = None  # module-level dataset fixture

_BUILDING_ID = 'sb1'
_PARTITION_IDS = ['2022_a', '2022_b', '2023_a', '2023_b', '2024_a']

_ACTION_IDS_MAP = {
    '12945159110931775488@supply_air_temperature_setpoint': 0,
    '13761436543392677888@supply_water_temperature_setpoint': 1,
    '14409954889734029312@supply_air_temperature_setpoint': 2,
}
_ACTION_IDS = list(_ACTION_IDS_MAP.keys())

_DEVICE_LAYOUT_IDS = [
    'VAV CO 1-1-06',
    'VAV CO 1-1-07 CO2',
    'VAV CO 1-1-08 CO2',
    'VAV CO 1-1-10 CO2',
    'VAV CO 1-1-13 CO2',
    'VAV CO 1-1-16 CO2',
    'VAV CO 1-1-17 CO2',
    'VAV CO 1-1-18 CO2',
    'VAV CO 1-1-26',
    'VAV CO 1-1-27',
    'VAV CO 1-1-35 CO2',
    'VAV CO 1-1-43',
    'VAV CO 1-1-51',
    'VAV RH 1-1-01',
    'VAV RH 1-1-02',
    'VAV RH 1-1-03',
    'VAV RH 1-1-04',
    'VAV RH 1-1-05',
    'VAV RH 1-1-09 CO2',
    'VAV RH 1-1-11',
    'VAV RH 1-1-12 CO2',
    'VAV RH 1-1-14 CO2',
    'VAV RH 1-1-15',
    'VAV RH 1-1-19',
    'VAV RH 1-1-20',
    'VAV RH 1-1-21',
    'VAV RH 1-1-22',
    'VAV RH 1-1-23',
    'VAV RH 1-1-25 (MK # 1F9)',
    'VAV RH 1-1-28 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-29 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-30 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-31 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-32 CO2 (Hearty Tech Talk 1H2)',
    'VAV RH 1-1-33',
    'VAV RH 1-1-34',
    'VAV RH 1-1-36',
    'VAV RH 1-1-37',
    'VAV RH 1-1-38',
    'VAV RH 1-1-39',
    'VAV RH 1-1-40',
    'VAV RH 1-1-41',
    'VAV RH 1-1-42',
    'VAV RH 1-1-44',
    'VAV RH 1-1-45',
    'VAV RH 1-1-46',
    'VAV RH 1-1-47',
    'VAV RH 1-1-48',
    'VAV RH 1-1-49',
    'VAV RH 1-1-50',
    'VAV RH 1-1-52 CO2',
    'VAV RH 1-1-53',
    'VAV RH 1-1-54',
    'VAV RH 1-1-55',
]

_FIRST_ZONE_INFO = {
    'zone_id': 'rooms/1002000133978',
    'building_id': 'buildings/3616672508',
    'zone_description': 'SB1-2-C2054',
    'area': 0.0,
    'zone_type': 1,
    'floor': 2,
    'devices': ['2618581107144046', '2696593986887004'],
}

_LAST_ZONE_INFO = {
    'zone_id': 'rooms/11312312488',
    'building_id': 'buildings/3616672508',
    'zone_description': 'SB1-1-1J7B',
    'area': 0.0,
    'zone_type': 1,
    'floor': 1,
    'devices': ['2802781341872564'],
}

_FIRST_DEVICE_INFO = {
    'device_id': '202194278473007104',
    'namespace': 'PHRED',
    'code': 'SB1:AHU:AC-2',
    'zone_id': '',
    'device_type': 6,
    'observable_fields': {
        'building_air_static_pressure_sensor': 1,
        'outside_air_flowrate_sensor': 1,
        'supply_fan_speed_percentage_command': 1,
        'supply_air_temperature_sensor': 1,
        'supply_fan_speed_frequency_sensor': 1,
        'supply_air_static_pressure_setpoint': 1,
        'return_air_temperature_sensor': 1,
        'mixed_air_temperature_setpoint': 1,
        'exhaust_fan_speed_percentage_command': 1,
        'exhaust_fan_speed_frequency_sensor': 1,
        'outside_air_damper_percentage_command': 1,
        'mixed_air_temperature_sensor': 1,
        'exhaust_air_damper_percentage_command': 1,
        'cooling_percentage_command': 1,
        'outside_air_flowrate_setpoint': 1,
        'supply_air_temperature_setpoint': 1,
        'building_air_static_pressure_setpoint': 1,
        'supply_air_static_pressure_sensor': 1,
    },
    'actionable_fields': {
        'exhaust_air_damper_percentage_command': 1,
        'supply_air_temperature_setpoint': 1,
        'supply_fan_speed_percentage_command': 1,
        'outside_air_flowrate_setpoint': 1,
        'cooling_percentage_command': 1,
        'mixed_air_temperature_setpoint': 1,
        'exhaust_fan_speed_percentage_command': 1,
        'outside_air_damper_percentage_command': 1,
        'supply_air_static_pressure_setpoint': 1,
        'building_air_static_pressure_setpoint': 1,
    },
}

_LAST_DEVICE_INFO = {
    'device_id': '2640423556868160',
    'namespace': 'CDM',
    'code': 'VAV RH 2-2-68',
    'zone_id': '',
    'device_type': 4,
    'observable_fields': {
        'supply_air_flowrate_setpoint': 1,
        'discharge_air_temperature_setpoint': 1,
        'discharge_air_temperature_sensor': 1,
        'zone_air_heating_temperature_setpoint': 1,
        'heating_water_valve_percentage_command': 1,
        'supply_air_flowrate_sensor': 1,
        'zone_air_temperature_sensor': 1,
        'zone_air_cooling_temperature_setpoint': 1,
        'supply_air_damper_percentage_command': 1,
    },
    'actionable_fields': {
        'zone_air_cooling_temperature_setpoint': 1,
        'discharge_air_temperature_setpoint': 1,
        'heating_water_valve_percentage_command': 1,
        'supply_air_flowrate_setpoint': 1,
        'zone_air_heating_temperature_setpoint': 1,
        'supply_air_damper_percentage_command': 1,
    },
}

_DEVICE_OBSERVABLE_FIELD_NAMES = [
    'building_air_static_pressure_sensor',
    'building_air_static_pressure_setpoint',
    'cooling_percentage_command',
    'differential_pressure_sensor',
    'differential_pressure_setpoint',
    'discharge_air_temperature_sensor',
    'discharge_air_temperature_setpoint',
    'exhaust_air_damper_percentage_command',
    'exhaust_air_damper_percentage_sensor',
    'exhaust_fan_speed_frequency_sensor',
    'exhaust_fan_speed_percentage_command',
    'heating_water_valve_percentage_command',
    'mixed_air_temperature_sensor',
    'mixed_air_temperature_setpoint',
    'outside_air_damper_percentage_command',
    'outside_air_dewpoint_temperature_sensor',
    'outside_air_flowrate_sensor',
    'outside_air_flowrate_setpoint',
    'outside_air_relative_humidity_sensor',
    'outside_air_specificenthalpy_sensor',
    'outside_air_temperature_sensor',
    'outside_air_wetbulb_temperature_sensor',
    'program_differential_pressure_setpoint',
    'program_supply_air_static_pressure_setpoint',
    'program_supply_air_temperature_setpoint',
    'program_supply_water_temperature_setpoint',
    'return_air_temperature_sensor',
    'return_water_temperature_sensor',
    'run_status',
    'speed_frequency_sensor',
    'speed_percentage_command',
    'supervisor_supply_air_static_pressure_setpoint',
    'supervisor_supply_air_temperature_setpoint',
    'supervisor_supply_water_temperature_setpoint',
    'supply_air_damper_percentage_command',
    'supply_air_flowrate_sensor',
    'supply_air_flowrate_setpoint',
    'supply_air_static_pressure_sensor',
    'supply_air_static_pressure_setpoint',
    'supply_air_temperature_sensor',
    'supply_air_temperature_setpoint',
    'supply_fan_run_status',
    'supply_fan_speed_frequency_sensor',
    'supply_fan_speed_percentage_command',
    'supply_water_temperature_sensor',
    'supply_water_temperature_setpoint',
    'zone_air_co2_concentration_sensor',
    'zone_air_co2_concentration_setpoint',
    'zone_air_cooling_temperature_setpoint',
    'zone_air_heating_temperature_setpoint',
    'zone_air_temperature_sensor',
]

_DEVICE_ACTIONABLE_FIELD_NAMES = [
    'building_air_static_pressure_setpoint',
    'cooling_percentage_command',
    'differential_pressure_setpoint',
    'discharge_air_temperature_setpoint',
    'exhaust_air_damper_percentage_command',
    'exhaust_fan_speed_percentage_command',
    'heating_water_valve_percentage_command',
    'mixed_air_temperature_setpoint',
    'outside_air_damper_percentage_command',
    'outside_air_flowrate_setpoint',
    'program_differential_pressure_setpoint',
    'program_supply_air_static_pressure_setpoint',
    'program_supply_air_temperature_setpoint',
    'program_supply_water_temperature_setpoint',
    'speed_percentage_command',
    'supervisor_supply_air_static_pressure_setpoint',
    'supervisor_supply_air_temperature_setpoint',
    'supervisor_supply_water_temperature_setpoint',
    'supply_air_damper_percentage_command',
    'supply_air_flowrate_setpoint',
    'supply_air_static_pressure_setpoint',
    'supply_air_temperature_setpoint',
    'supply_fan_speed_percentage_command',
    'supply_water_temperature_setpoint',
    'zone_air_co2_concentration_setpoint',
    'zone_air_cooling_temperature_setpoint',
    'zone_air_heating_temperature_setpoint',
]

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

#
# SET UP METHODS
#


def cleanup_files():
  print('Deleting dataset files...')

  if os.path.isfile(ZIP_FILEPATH):
    os.remove(ZIP_FILEPATH)

  if os.path.isdir(DATASET_DIRPATH):
    shutil.rmtree(DATASET_DIRPATH)


def setUpModule():
  """Module-level setup. Cleans up files as desired. Downloads data as desired.
  Initializes the dataset as a module level fixture as desired.
  """
  global _dataset

  if TEST_DATASET_DOWNLOAD and CLEAR_TEST_DATASET_DOWNLOAD:
    cleanup_files()

  print('Initializing BuildingDataset (this should happen only once)...')
  _dataset = BuildingDataset(building_id='sb1', download=TEST_DATASET_DOWNLOAD)


#
# TESTS
#


class TestDataDirectory(absltest.TestCase):
  """Tests for the data directory."""

  def test_data_dir(self):
    self.assertTrue(os.path.isdir(DATA_DIR))


class TestBuildingDataset(absltest.TestCase):
  """Tests for the BuildingDataset class."""

  ds = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.ds = _dataset

  def test_building_id(self):
    self.assertEqual(self.ds.building_id, _BUILDING_ID)

  def test_partition_ids(self):
    self.assertEqual(self.ds.partition_ids, _PARTITION_IDS)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_download(self):
    self.assertIn('download', dir(self.ds))
    self.assertTrue(os.path.isdir(DATASET_DIRPATH))
    self.assertTrue(os.path.exists(ZIP_FILEPATH))

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_floorplan(self):
    floorplan = self.ds.floorplan

    self.assertIsInstance(floorplan, np.ndarray)
    self.assertEqual(floorplan.shape, (744, 1004))

    values, counts = np.unique(floorplan, return_counts=True)
    value_counts = dict(zip(values, counts))
    self.assertEqual(value_counts, {0.0: 436332, 1.0: 60204, 2.0: 250440})

  # DEVICES

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_device_layout_map(self):
    device_layout_map = self.ds.device_layout_map

    self.assertIsInstance(device_layout_map, dict)
    self.assertEqual(sorted(list(device_layout_map.keys())), _DEVICE_LAYOUT_IDS)

    # each device layout is a list of lists containing two integer coordinates.
    # layouts may differ in length:

    example_layout_map = device_layout_map['VAV CO 1-1-06']
    self.assertIsInstance(example_layout_map, list)
    self.assertEqual(np.array(example_layout_map).shape, (1021, 2))
    self.assertEqual(example_layout_map[0], [79, 35])
    self.assertEqual(example_layout_map[-1], [80, 64])

    another_layout_map = device_layout_map['VAV RH 1-1-55']
    self.assertIsInstance(another_layout_map, list)
    self.assertEqual(np.array(another_layout_map).shape, (935, 2))
    self.assertEqual(another_layout_map[0], [145, 126])
    self.assertEqual(another_layout_map[-1], [149, 160])

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_device_infos(self):
    device_infos = self.ds.device_infos
    self.assertIsInstance(device_infos, list)
    self.assertEqual(len(device_infos), 173)
    self.assertEqual(device_infos[0], _FIRST_DEVICE_INFO)
    self.assertEqual(device_infos[-1], _LAST_DEVICE_INFO)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_devices_df(self):
    devices_df = self.ds.devices_df

    self.assertIsInstance(devices_df, pd.DataFrame)
    self.assertEqual(len(devices_df), 173)

    # each row uniquely identified by the device identifier:
    self.assertEqual(devices_df['device_id'].nunique(), len(devices_df))

    self.assertEqual(
        devices_df.columns.tolist(),
        [
            'device_id',
            'namespace',
            'code',
            'device_type',
            'observable_fields',
            'actionable_fields',
        ],
    )

    first_row = _FIRST_DEVICE_INFO.copy()
    del first_row['zone_id']  # we removed "zone_id" from the df
    self.assertEqual(devices_df.iloc[0].to_dict(), first_row)

    # each device belongs to a namespace:
    self.assertEqual(
        devices_df['namespace'].value_counts().to_dict(),
        {'CDM': 155, 'PHRED': 18},
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_observable_fields(self):
    observable_fields = self.ds.observable_fields
    self.assertIsInstance(observable_fields, list)
    self.assertEqual(len(observable_fields), 51)
    self.assertEqual(observable_fields, _DEVICE_OBSERVABLE_FIELD_NAMES)

    value_counts = self.ds.observable_field_counts
    self.assertIsInstance(value_counts, pd.Series)
    self.assertEqual(len(value_counts), 51)
    self.assertEqual(
        value_counts.head(1).to_dict(),
        {'supply_air_damper_percentage_command': 123},
    )

    self.assertEqual(
        value_counts.tail(1).to_dict(),
        {'supervisor_supply_water_temperature_setpoint': 1},
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_actionable_fields(self):
    actionable_fields = self.ds.actionable_fields
    self.assertIsInstance(actionable_fields, list)
    self.assertEqual(len(actionable_fields), 27)
    self.assertEqual(actionable_fields, _DEVICE_ACTIONABLE_FIELD_NAMES)

    value_counts = self.ds.observable_field_counts
    self.assertIsInstance(value_counts, pd.Series)
    self.assertEqual(len(value_counts), 51)
    self.assertEqual(
        value_counts.head(1).to_dict(),
        {'supply_air_damper_percentage_command': 123},
    )
    self.assertEqual(
        value_counts.tail(1).to_dict(),
        {'supervisor_supply_water_temperature_setpoint': 1},
    )

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_fields_df(self):
    fields_df = self.ds.fields_df
    self.assertIsInstance(fields_df, pd.DataFrame)
    self.assertEqual(len(fields_df), 51)
    self.assertEqual(
        fields_df.columns.tolist(),
        [
            'field_name',
            'is_actionable',
            'is_observable',
            'devices_actionable',
            'devices_observable',
        ],
    )
    # some are observable:
    self.assertEqual(
        fields_df.iloc[0].to_dict(),
        {
            'field_name': 'building_air_static_pressure_sensor',
            'is_actionable': False,
            'is_observable': True,
            'devices_actionable': 0,
            'devices_observable': 3,
        },
    )
    # some are actionable:
    self.assertEqual(
        fields_df.iloc[-1].to_dict(),
        {
            'field_name': 'zone_air_temperature_sensor',
            'is_actionable': False,
            'is_observable': True,
            'devices_actionable': 0,
            'devices_observable': 123,
        },
    )
    # some are both actionable and observable:
    self.assertEqual(
        len(fields_df[(fields_df.is_actionable & fields_df.is_observable)]), 27
    )

  # ZONES

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_zone_infos(self):
    zone_infos = self.ds.zone_infos

    self.assertIsInstance(zone_infos, list)
    self.assertEqual(len(zone_infos), 563)

    self.assertEqual(zone_infos[0], _FIRST_ZONE_INFO)
    self.assertEqual(zone_infos[-1], _LAST_ZONE_INFO)

  @unittest.skipUnless(TEST_DATASET, SKIP_REASON)
  def test_zones_df(self):
    zones_df = self.ds.zones_df

    self.assertIsInstance(zones_df, pd.DataFrame)
    self.assertEqual(len(zones_df), 563)

    # each row uniquely identified by the zone identifier:
    self.assertEqual(zones_df['zone_id'].nunique(), len(zones_df))

    self.assertEqual(
        zones_df.columns.tolist(),
        [
            'zone_id',
            'building_id',
            'zone_description',
            'area',
            'zone_type',
            'floor',
            'devices',
            'n_devices',
        ],
    )

    first_row = _FIRST_ZONE_INFO.copy()
    first_row['n_devices'] = 2  # we added this column to the df
    self.assertEqual(zones_df.iloc[0].to_dict(), first_row)


class TestBuildingDatasetPartition(absltest.TestCase):
  """Tests for the BuildingDatasetPartition class."""

  ds = None
  partition = None

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.ds = _dataset
    cls.partition = BuildingDatasetPartition(
        building_dataset=cls.ds, partition_id='2022_a'
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

    # we are surfacing each key into its own high-level public property
    # fmt: off
    # pylint: disable=line-too-long
    with self.subTest('action_value_matrix'):
      np.testing.assert_array_equal(
          data['action_value_matrix'], self.partition.action_value_matrix
      )
    with self.subTest('observation_value_matrix'):
      np.testing.assert_array_equal(
          data['observation_value_matrix'], self.partition.observation_value_matrix
      )
    with self.subTest('reward_value_matrix'):
      np.testing.assert_array_equal(
          data['reward_value_matrix'], self.partition.reward_value_matrix
      )
    with self.subTest('reward_info_value_matrix'):
      np.testing.assert_array_equal(
          data['reward_info_value_matrix'], self.partition.reward_info_value_matrix
      )
    # pylint: enable=line-too-long
    # fmt: on

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

    # keys are in the format of `device_id@setting_name`:
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

    # keys are in the format of `zone_id@setting` or `device_id@setting`:
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

    # values are numeric (float) and non-null:
    self.assertEqual(df.isna().sum().sum(), 0)
    self.assertEqual(df.dtypes.unique().tolist(), [np.dtype('float64')])


if __name__ == '__main__':
  absltest.main()
