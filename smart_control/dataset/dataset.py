"""Smart Buildings Dataset implementation, including loading and downloading."""

import json
import os
import pickle
import shutil

import numpy as np
import requests

from smart_control.utils.constants import ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")


class SmartBuildingsDataset:
  """Smart Buildings Dataset."""

  def __init__(self, download=True):
    self.partitions = {
        "sb1": [
            "2022_a",
            "2022_b",
            "2023_a",
            "2023_b",
            "2024_a",
        ],
    }
    if download:
      self.download()

  @staticmethod
  def _download_file_if_not_exists(url, timeout=60):
    local_filename = url.split("/")[-1]
    local_filepath = os.path.join(DATA_DIR, local_filename)

    if not os.path.isfile(local_filepath):

      with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        with open(local_filepath, "wb") as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_filepath

  def download(self):
    """Downloads the Smart Buildings Dataset from Google Cloud Storage."""
    print("Downloading data...")
    url = "https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip"  # pylint: disable=line-too-long
    self._download_file_if_not_exists(url)

    zip_filepath = os.path.join(DATA_DIR, "sb1.zip")
    dataset_dir = os.path.join(DATA_DIR, "sb1/")
    shutil.unpack_archive(zip_filepath, dataset_dir)

  def get_floorplan(self, building):
    """Gets the floorplan and device layout map for a specific building.

    Args:
      building: The name of the building.

    Returns:
      A tuple containing the floorplan and device layout map.
    """
    if building not in self.partitions:
      raise ValueError("Invalid building")

    floorplan = np.load(f"./{building}/tabular/floorplan.npy")

    json_filepath = f"./{building}/tabular/device_layout_map.json"
    with open(json_filepath, encoding="utf-8") as json_file:
      device_layout_map = json.load(json_file)

    return floorplan, device_layout_map

  def get_building_data(self, building, partition):
    """Gets the data for a specific building and partition.

    Args:
      building: The name of the building.
      partition: The name of the partition.

    Returns:
      A tuple containing the data and metadata.
    """
    if building not in self.partitions:
      raise ValueError("Invalid building")

    if partition not in self.partitions[building]:
      raise ValueError("invalid partition")

    path = f"./{building}/tabular/{building}/{partition}/"

    data = np.load(path + "data.npy.npz")
    metadata = pickle.load(open(path + "metadata.pickle", "rb"))

    if "device_infos" not in metadata.keys():
      metadata["device_infos"] = pickle.load(
          open(f"./{building}/tabular/device_info_dicts.pickle", "rb")
      )
    if "zone_infos" not in metadata.keys():
      metadata["zone_infos"] = pickle.load(
          open(f"./{building}/tabular/zone_info_dicts.pickle", "rb")
      )
    return data, metadata
