"""Tests for SmartBuildingsDataset"""

import os
import shutil
import unittest

from absl.testing import absltest
from dotenv import load_dotenv

from smart_control.dataset.dataset import DATA_DIR
from smart_control.dataset.dataset import SmartBuildingsDataset

load_dotenv()

TEST_DATASET_DOWNLOAD = bool(os.getenv("TEST_DATASET_DOWNLOAD", default="false").lower() == "true")  # pylint: disable=line-too-long


class TestDataset(absltest.TestCase):
  """Tests for the Smart Buildings Dataset"""

  def test_data_dir(self):
    self.assertTrue(os.path.isdir(DATA_DIR))

  @unittest.skipUnless(TEST_DATASET_DOWNLOAD, "Skip large download by default.")
  def test_download(self):
    # NOTE: currently this is an end-to-end test, but takes around 2 minutes.
    # We are skipping by default to keep the build fast.
    # If you would like to run it periodically, locally:
    # set the `TEST_DATASET_DOWNLOAD` env var to "true"
    # TODO: consider adding a mocked version of the test as well

    dataset_dirpath = os.path.join(DATA_DIR, "sb1")
    zip_filepath = os.path.join(DATA_DIR, "sb1.zip")

    shutil.rmtree(dataset_dirpath)
    if os.path.exists(zip_filepath):
      os.remove(zip_filepath)
    self.assertFalse(os.path.isdir(dataset_dirpath))
    self.assertFalse(os.path.exists(zip_filepath))

    ds = SmartBuildingsDataset()
    ds.download()

    self.assertTrue(os.path.isdir(dataset_dirpath))
    self.assertTrue(os.path.exists(zip_filepath))

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


if __name__ == "__main__":
  absltest.main()
