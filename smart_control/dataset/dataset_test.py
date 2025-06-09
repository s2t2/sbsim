"""Tests for SmartBuildingsDataset"""

import os
import shutil
import unittest

from absl.testing import absltest
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from smart_control.dataset.dataset import DATA_DIR
from smart_control.dataset.dataset import SmartBuildingsDataset

load_dotenv()

TEST_DATASET_DOWNLOAD = bool(os.getenv('TEST_DATASET_DOWNLOAD', default='false').lower() == 'true')  # pylint: disable=line-too-long
CLEAR_DATASET_DOWNLOAD = bool(os.getenv('CLEAR_DATASET_DOWNLOAD', default='false').lower() == 'true')  # pylint: disable=line-too-long

DATASET_DIRPATH = os.path.join(DATA_DIR, 'sb1')
ZIP_FILEPATH = os.path.join(DATA_DIR, 'sb1.zip')

SKIP_REASON = 'Skip large download by default.'


class TestDataset(absltest.TestCase):
  """Tests for the Smart Buildings Dataset.

  Includes some high fidelity tests to download the actual dataset.

  It takes around two minutes to download the data, so we are skipping certain
  tests by default, to keep the build fast. But you can run them manually by
  setting the `TEST_DATASET_DOWNLOAD` environment variable to 'true'.

  Downloaded data will not get cleared by default, but you can force a clean up
  and fresh download by setting the `CLEAR_DATASET_DOWNLOAD` environment
  variable to 'true'.

  These tests are meant to be run periodically, for example once per day.
  """

  ds = None

  @staticmethod
  def _cleanup_files():
    print('Deleting dataset files...')

    if os.path.exists(ZIP_FILEPATH):
      os.remove(ZIP_FILEPATH)

    if os.path.exists(DATASET_DIRPATH):
      shutil.rmtree(DATASET_DIRPATH)

  def _assert_timestamps(self, timestamps, earliest, latest, length):
    """
    Custom assertion for standardized timestamp format.

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
    self.assertIsInstance(first_timestamp, pd._libs.tslibs.timestamps.Timestamp)
    self.assertIsInstance(last_timestamp, pd._libs.tslibs.timestamps.Timestamp)
    self.assertEqual(str(first_timestamp), earliest)
    self.assertEqual(str(last_timestamp), latest)

  @classmethod
  def setUpClass(cls):
    if TEST_DATASET_DOWNLOAD:
      if CLEAR_DATASET_DOWNLOAD:
        cls._cleanup_files()
      cls.ds = SmartBuildingsDataset(download=True)
    else:
      cls.ds = SmartBuildingsDataset(download=False)

  @classmethod
  def tearDownClass(cls):
    if TEST_DATASET_DOWNLOAD and CLEAR_DATASET_DOWNLOAD:
      cls._cleanup_files()

  def test_data_dir(self):
    self.assertTrue(os.path.isdir(DATA_DIR))

  def test_partitions(self):
    partitions = {'sb1': ['2022_a', '2022_b', '2023_a', '2023_b', '2024_a']}
    self.assertEqual(self.ds.partitions, partitions)

  @unittest.skipUnless(TEST_DATASET_DOWNLOAD, SKIP_REASON)
  def test_download(self):
    self.assertTrue(os.path.isdir(DATASET_DIRPATH))
    self.assertTrue(os.path.exists(ZIP_FILEPATH))

  @unittest.skipUnless(TEST_DATASET_DOWNLOAD, SKIP_REASON)
  def test_get_floorplan(self):
    ds = self.ds

    floorplan, device_layout_map = ds.get_floorplan('sb1')

    # FLOORPLAN

    self.assertIsInstance(floorplan, np.ndarray)
    self.assertEqual(floorplan.shape, (744, 1004))
    values, counts = np.unique(floorplan, return_counts=True)
    value_counts = dict(zip(values, counts))
    self.assertEqual(value_counts, {0.0: 436332, 1.0: 60204, 2.0: 250440})

    # DEVICE LAYOUT MAP

    self.assertIsInstance(device_layout_map, dict)
    device_keys = sorted(list(device_layout_map.keys()))
    expected_device_keys = [
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
    self.assertEqual(device_keys, expected_device_keys)
    # each device layout map is a list of two integer values:
    first_layout_map = device_layout_map['VAV CO 1-1-06']
    self.assertIsInstance(first_layout_map, list)
    self.assertEqual(np.array(first_layout_map).shape, (1021, 2))
    self.assertEqual(first_layout_map[0], [79, 35])
    self.assertEqual(first_layout_map[-1], [80, 64])

  @unittest.skipUnless(TEST_DATASET_DOWNLOAD, SKIP_REASON)
  def test_get_building_data(self):
    ds = self.ds

    data, metadata = ds.get_building_data('sb1', '2022_a')

    #
    # DATA
    #

    self.assertIsInstance(data, np.lib.npyio.NpzFile)
    self.assertEqual(data['observation_value_matrix'].shape, (51852, 1198))
    self.assertEqual(data['action_value_matrix'].shape, (51852, 3))
    self.assertEqual(data['reward_value_matrix'].shape, (51852, 17))
    self.assertEqual(data['reward_info_value_matrix'].shape, (51852, 3252))

    #
    # METADATA
    #

    self.assertIsInstance(metadata, dict)
    metadata_keys = [
        'action_ids',
        'action_timestamps',
        'device_infos',
        'observation_ids',
        'observation_timestamps',
        'reward_ids',
        'reward_info_timestamps',
        'reward_timestamps',
        'zone_infos',
    ]
    self.assertEqual(sorted(metadata.keys()), metadata_keys)

    # action_timestamps:
    self._assert_timestamps(
        metadata['action_timestamps'],
        earliest='2022-01-01 00:00:00+00:00',
        latest='2022-06-30 00:55:00+00:00',
        length=51852,
    )

    # observation_timestamps:
    self._assert_timestamps(
        metadata['observation_timestamps'],
        earliest='2022-01-01 00:00:00+00:00',
        latest='2022-06-30 00:55:00+00:00',
        length=51852,
    )

    # reward_info_timestamps:
    self._assert_timestamps(
        metadata['reward_info_timestamps'],
        earliest='2021-12-31 23:55:00+00:00',
        latest='2022-06-30 00:50:00+00:00',
        length=51852,
    )

    # reward_timestamps:
    self._assert_timestamps(
        metadata['reward_timestamps'],
        earliest='2021-12-31 23:55:00+00:00',
        latest='2022-06-30 00:50:00+00:00',
        length=51852,
    )

    # action_ids:
    action_ids = {
        '12945159110931775488@supply_air_temperature_setpoint': 0,
        '13761436543392677888@supply_water_temperature_setpoint': 1,
        '14409954889734029312@supply_air_temperature_setpoint': 2,
    }
    self.assertEqual(metadata['action_ids'], action_ids)

    # device_infos:
    self.assertIsInstance(metadata['device_infos'], list)
    self.assertEqual(len(metadata['device_infos']), 173)
    first_device_info = {
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
        'action_fields': {
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
    self.assertEqual(metadata['device_infos'][0], first_device_info)

    # observation_ids:
    self.assertIsInstance(metadata['observation_ids'], dict)
    self.assertEqual(len(metadata['observation_ids']), 1198)
    # an example key / value pair that exists:
    self.assertEqual(
        metadata['observation_ids']['202194278473007104@building_air_static_pressure_setpoint'],  # pylint:disable=line-too-long
        0,
    )

    # reward_ids:
    self.assertIsInstance(metadata['reward_ids'], dict)
    self.assertEqual(len(metadata['reward_ids']), 3252)
    # an example key / value pair that exists:
    self.assertEqual(
        metadata['reward_ids']['rooms/9028552126@heating_setpoint_temperature'],
        0,
    )

    # 'zone_infos':
    self.assertIsInstance(metadata['zone_infos'], list)
    self.assertEqual(len(metadata['zone_infos']), 563)
    first_zone_info = {
        'zone_id': 'rooms/1002000133978',
        'building_id': 'buildings/3616672508',
        'zone_description': 'SB1-2-C2054',
        'area': 0.0,
        'zone_type': 1,
        'floor': 2,
        'devices': ['2618581107144046', '2696593986887004'],
    }
    self.assertEqual(metadata['zone_infos'][0], first_zone_info)


if __name__ == '__main__':
  absltest.main()
