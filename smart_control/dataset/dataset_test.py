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

TEST_DATASET_DOWNLOAD = bool(os.getenv('TEST_DATASET_DOWNLOAD', default='false').lower() == 'true') # pylint: disable=line-too-long

DATASET_DIRPATH = os.path.join(DATA_DIR, 'sb1')
ZIP_FILEPATH = os.path.join(DATA_DIR, 'sb1.zip')

SKIP_REASON = 'Skip large download by default.'


def cleanup_files():
  if os.path.exists(ZIP_FILEPATH):
    os.remove(ZIP_FILEPATH)

  if os.path.exists(DATASET_DIRPATH):
    shutil.rmtree(DATASET_DIRPATH)


class TestDataDirectory(absltest.TestCase):
  """Tests for the data directory."""

  def test_data_dir(self):
    self.assertTrue(os.path.isdir(DATA_DIR))


class TestDataset(absltest.TestCase):
  """Tests for the Smart Buildings Dataset, using real downloaded data.

  It takes around two minutes to download the data, so we are skipping these
  tests by default, to keep the build fast. But you can run them manually by
  setting the `TEST_DATASET_DOWNLOAD` environment variable to 'true'.

  TODO: consider adding a mocked variation as well.
  """

  ds = None

  @classmethod
  def setUpClass(cls):
    if TEST_DATASET_DOWNLOAD:
      cleanup_files()
      assert not os.path.isdir(DATASET_DIRPATH)
      assert not os.path.exists(ZIP_FILEPATH)

      cls.ds = SmartBuildingsDataset(download=True)

      assert os.path.isdir(DATASET_DIRPATH)
      assert os.path.exists(ZIP_FILEPATH)

  #@classmethod
  #def tearDownClass(cls):
  #  if TEST_DATASET_DOWNLOAD:
  #    cleanup_files()
  #    assert not os.path.isdir(DATASET_DIRPATH)
  #    assert not os.path.exists(ZIP_FILEPATH)

  @unittest.skipUnless(TEST_DATASET_DOWNLOAD, SKIP_REASON)
  def test_download(self):
    self.assertTrue(os.path.isdir(DATASET_DIRPATH))
    self.assertTrue(os.path.exists(ZIP_FILEPATH))

  @unittest.skipUnless(TEST_DATASET_DOWNLOAD, SKIP_REASON)
  def test_get_floorplan(self):
    ds = self.ds

    floorplan, device_layout_map = ds.get_floorplan('sb1')

    # TODO:
    #breakpoint()
    print(type(floorplan), type(device_layout_map))

    #self.assertIsInstance(floorplan, np.ndarray)
    #self.assertIsInstance(device_layout_map, dict)
    #self.assertGreater(floorplan.shape[0], 0)
    #self.assertIn('device_name_example', device_layout_map)

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
    metadata_keys = ['action_ids', 'action_timestamps', 'device_infos',
                     'observation_ids', 'observation_timestamps', 'reward_ids',
                     'reward_info_timestamps', 'reward_timestamps', 'zone_infos'
                     ]
    self.assertEqual(sorted(metadata.keys()), metadata_keys)

    # action_ids:
    action_ids = {
      '12945159110931775488@supply_air_temperature_setpoint': 0,
      '13761436543392677888@supply_water_temperature_setpoint': 1,
      '14409954889734029312@supply_air_temperature_setpoint': 2
    }
    self.assertEqual(metadata['action_ids'], action_ids)

    # action_timestamps:
    self.assertEqual(len(metadata['action_timestamps']), 173)
    first_timestamp = metadata['action_timestamps'][0]
    last_timestamp = metadata['action_timestamps'][-1]
    self.assertIsInstance(first_timestamp, pd._libs.tslibs.timestamps.Timestamp)
    self.assertIsInstance(last_timestamp, pd._libs.tslibs.timestamps.Timestamp)
    self.assertEqual(str(first_timestamp), '2022-01-01 00:00:00+00:00')
    self.assertEqual(str(last_timestamp),  '2022-06-30 00:55:00+00:00')

    # device_infos:
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
        'supply_air_static_pressure_sensor': 1
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
        'building_air_static_pressure_setpoint': 1
      }
    }
    self.assertEqual(metadata['device_infos'][0], first_device_info)

<<<<<<< HEAD
  # def test_get_floorplan(self):
  #  ds = SmartBuildingsDataset()
  #
  #  result = ds.get_floorplan("sb1")
  #  self.assertEqual()

  # def test_get_building_data(self):
  #  ds = SmartBuildingsDataset()
  #
  #  result = get_building_data("sb1", "2022_a")
  #  self.assertEqual()
=======

    # observation_ids:


    # observation_timestamps:
>>>>>>> c6dab15 (Dataset tests - WIP)


    # reward_ids:


    # reward_info_timestamps:


    # reward_timestamps:


    # 'zone_infos':




if __name__ == '__main__':
  absltest.main()
