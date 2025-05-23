"""Manages the Smart Buildings Dataset, handling downloads and data access.

This module provides the SmartBuildingsDataset class, which facilitates
downloading the dataset from Google Cloud Storage and accessing its various
components, such as floorplans and building data.
"""

import json
import pickle
import shutil
import numpy as np
import requests


class SmartBuildingsDataset:
  """Manages the Smart Buildings Dataset.

  This class handles downloading the dataset from Google Cloud Storage and
  provides methods to access floorplans, device layouts, and time-series
  data for specified buildings and partitions.

  Attributes:
    partitions (dict): A dictionary mapping building names to a list of
      available data partitions (e.g., "2022_a", "2023_b").

  Examples:
    >>> dataset = SmartBuildingsDataset()
    >>> floorplan, device_layout = dataset.get_floorplan("sb1")
    >>> data, metadata = dataset.get_building_data("sb1", "2022_a")
  """

  def __init__(self, download: bool = True):
    """Initializes the SmartBuildingsDataset.

    Args:
      download (bool): If True (default), downloads the dataset upon
        initialization.
    """
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

  def download(self):
    """Downloads and unpacks the Smart Buildings Dataset.

    The dataset is downloaded from a Google Cloud Storage URL as a ZIP file
    (sb1.zip) and then unpacked into a directory named "sb1/" in the current
    working directory.
    """
    print("Downloading data...")

    def download_file(url: str) -> str:
      """Downloads a file from a URL to the local current directory.

      Args:
        url (str): The URL of the file to download.

      Returns:
        str: The local filename of the downloaded file.

      Raises:
        requests.exceptions.HTTPError: If the download fails.
      """
      local_filename = url.split("/")[-1]
      with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
      return local_filename

    url = (
        "https://storage.googleapis.com/gresearch/smart_buildings_dataset/"
        "tabular_data/sb1.zip"
    )
    download_file(url)
    shutil.unpack_archive("sb1.zip", "sb1/")

  def get_floorplan(self, building: str) -> tuple[np.ndarray, dict]:
    """Gets the floorplan and device layout map for a specific building.

    Args:
      building (str): The name of the building (e.g., "sb1").

    Returns:
      tuple:
        - np.ndarray: The floorplan layout as a NumPy array.
        - dict: The device layout map, where keys are device IDs and values
          are their coordinates or other layout information.

    Raises:
      ValueError: If the specified building is not found in the dataset.
    """
    if building not in self.partitions.keys():
      raise ValueError(f"Invalid building: {building}")
    floorplan_path = f"./{building}/tabular/floorplan.npy"
    device_map_path = f"./{building}/tabular/device_layout_map.json"

    floorplan = np.load(floorplan_path)
    with open(device_map_path) as json_file:
      device_layout_map = json.load(json_file)
    return floorplan, device_layout_map

  def get_building_data(
      self, building: str, partition: str
  ) -> tuple[np.ndarray, dict]:
    """Gets the time-series data and metadata for a building and partition.

    Args:
      building (str): The name of the building (e.g., "sb1").
      partition (str): The name of the data partition (e.g., "2022_a").

    Returns:
      tuple:
        - np.ndarray: The time-series data, typically stored in an .npz file.
        - dict: A dictionary containing metadata, including device information
          and zone information.

    Raises:
      ValueError: If the specified building or partition is not valid.
    """
    if building not in self.partitions.keys():
      raise ValueError(f"Invalid building: {building}")
    if partition not in self.partitions[building]:
      raise ValueError(f"Invalid partition: {partition} for {building}")

    base_path = f"./{building}/tabular/"
    partition_path = f"{base_path}{building}/{partition}/"

    data = np.load(partition_path + "data.npy.npz")
    with open(partition_path + "metadata.pickle", "rb") as f:
      metadata = pickle.load(f)

    if "device_infos" not in metadata:
      with open(base_path + "device_info_dicts.pickle", "rb") as f:
        metadata["device_infos"] = pickle.load(f)
    if "zone_infos" not in metadata:
      with open(base_path + "zone_info_dicts.pickle", "rb") as f:
        metadata["zone_infos"] = pickle.load(f)
    return data, metadata
