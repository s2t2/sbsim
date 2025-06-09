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

    if os.path.isfile(local_filepath):
      print("Using previously-downloaded data...")
      print(local_filepath)
    else:
      print("Downloading data...")
      print(url)
      with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(local_filepath, "wb") as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_filepath

  def download(self):
    """Downloads the Smart Buildings Dataset from Google Cloud Storage."""
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

    tabular_dirpath = os.path.join(DATA_DIR, building, "tabular")

    floorplan_filepath = os.path.join(tabular_dirpath, "floorplan.npy")
    floorplan = np.load(floorplan_filepath)

    device_layout_map_filepath = os.path.join(tabular_dirpath, "device_layout_map.json")  # pylint:disable=line-too-long
    with open(device_layout_map_filepath, encoding="utf-8") as json_file:
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
      raise ValueError(f"Invalid building: {building}")

    if partition not in self.partitions[building]:
      raise ValueError(f"Invalid partition: {partition}.")

    partition_dirpath = os.path.join(DATA_DIR, building, "tabular", building, partition)  # pylint:disable=line-too-long

    data_filepath = os.path.join(partition_dirpath, "data.npy.npz")
    data = np.load(data_filepath)

    metadata_filepath = os.path.join(partition_dirpath, "metadata.pickle")
    metadata = pickle.load(open(metadata_filepath, "rb"))

    if "device_infos" not in metadata.keys():
      device_info_filepath = os.path.join(DATA_DIR, building, "tabular", "device_info_dicts.pickle")  # pylint:disable=line-too-long
      metadata["device_infos"] = pickle.load(open(device_info_filepath, "rb"))

    if "zone_infos" not in metadata.keys():
      zone_info_filepath = os.path.join(DATA_DIR, building, "tabular", "zone_info_dicts.pickle")  # pylint:disable=line-too-long
      metadata["zone_infos"] = pickle.load(open(zone_info_filepath, "rb"))

    return data, metadata
