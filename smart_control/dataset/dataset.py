"""Smart Buildings Dataset implementation, including loading and downloading."""

import json
import pickle
import shutil
import numpy as np
import requests


class SmartBuildingsDataset:
  """Manages the Smart Buildings Dataset, including downloading and accessing data.

  This dataset contains sensor readings, building layout information, and other
  data relevant to smart building applications. It can be used for tasks such
  as building energy optimization, occupant comfort analysis, and fault detection.
  """

  def __init__(self, download=True):
    """Initializes the SmartBuildingsDataset object.

    Args:
      download: If True (default), downloads the dataset upon initialization.
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
    """Downloads the Smart Buildings Dataset from Google Cloud Storage.

    The dataset is downloaded as a ZIP file from
    https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip
    and extracted into the `./sb1/` directory.
    """
    print("Downloading data...")

    def download_file(url):
      local_filename = url.split("/")[-1]
      with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
      return local_filename

    url = "https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip"
    download_file(url)
    shutil.unpack_archive("sb1.zip", "sb1/")

  def get_floorplan(self, building):
    """Gets the floorplan and device layout map for a specific building.

    Args:
      building: The name of the building (e.g., "sb1").

    Returns:
      A tuple containing:
        - floorplan: A NumPy array representing the floorplan layout.
        - device_layout_map: A dictionary mapping device IDs to their
          locations and types.

    Example:
      >>> dataset = SmartBuildingsDataset(download=False)
      >>> floorplan, device_map = dataset.get_floorplan("sb1")
      >>> print(floorplan.shape)
      >>> print(type(device_map))
    """
    if building not in self.partitions.keys():
      raise ValueError("invalid building")
    floorplan = np.load(f"./{building}/tabular/floorplan.npy")
    with open(f"./{building}/tabular/device_layout_map.json") as json_file:
      device_layout_map = json.load(json_file)
    return floorplan, device_layout_map

  def get_building_data(self, building, partition):
    """Gets the data for a specific building and partition.

    Args:
      building: The name of the building (e.g., "sb1").
      partition: The name of the partition (e.g., "2022_a").

    Returns:
      A tuple containing:
        - data: A NumPy array or structured array containing the time-series
          sensor data.
        - metadata: A dictionary containing metadata associated with the
          partition, such as device information and zone information.

    Example:
      >>> dataset = SmartBuildingsDataset(download=False)
      >>> data, metadata = dataset.get_building_data("sb1", "2022_a")
      >>> print(data.keys())  # If data is a NpzFile
      >>> print(metadata.keys())
    """
    if building not in self.partitions.keys():
      raise ValueError("invalid building")
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
