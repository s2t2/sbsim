"""Smart Buildings Dataset implementation, including loading and downloading."""

from functools import cached_property
import json
import os
import pickle
import shutil

from matplotlib import pyplot as plt
import numpy as np
import requests

from smart_control.utils.constants import ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "data")

VALID_BUILDING_PARTITIONS = {
    "sb1": ["2022_a", "2022_b", "2023_a", "2023_b", "2024_a"]
}


class BuildingDataset:
  """A helper class for handling the dataset for a specific building.

  Args:
    building_id (str): The identifier of the building (e.g. "sb1").
    download (bool): Whether or not to download the dataset.
  """

  def __init__(self, building_id="sb1", download=True):
    self.building_id = building_id

    if self.building_id not in VALID_BUILDING_PARTITIONS:
      raise ValueError("Invalid building: '{self.building_id}'.")

    self.partition_ids = VALID_BUILDING_PARTITIONS[self.building_id]

    if bool(download):
      self.download()

  @property
  def zip_filename(self):
    return f"{self.building_id}.zip"

  @property
  def zip_url(self):
    return (
        "https://storage.googleapis.com/gresearch/smart_buildings_dataset/"
        f"tabular_data/{self.zip_filename}"
    )

  @property
  def zip_filepath(self):
    return os.path.join(DATA_DIR, self.zip_filename)

  @property
  def building_dirpath(self):
    return os.path.join(DATA_DIR, self.building_id)

  @staticmethod
  def _download_file_if_not_exists(url, timeout=60):
    local_filename = url.split("/")[-1]
    local_filepath = os.path.join(DATA_DIR, local_filename)

    if os.path.isfile(local_filepath):
      print("Using previously-downloaded data...")
      print(os.path.abspath(local_filepath))
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
    """Downloads the building's dataset from Google Cloud Storage."""
    self._download_file_if_not_exists(self.zip_url)
    shutil.unpack_archive(self.zip_filepath, self.building_dirpath)

  @property
  def tabular_dirpath(self):
    return os.path.join(self.building_dirpath, "tabular")

  @property
  def floorplan_filepath(self):
    return os.path.join(self.tabular_dirpath, "floorplan.npy")

  @cached_property
  def floorplan(self):
    """The building's floorplan."""
    return np.load(self.floorplan_filepath)

  @property
  def floorplan_image_filepath(self):
    floorplan_image_filename = f"{self.building_id}_floorplan.png"
    return os.path.join(self.building_dirpath, floorplan_image_filename)

  def display_floorplan(
      self, cmap="binary", show=True, save=True, image_filepath=None
  ):
    """Show an image of floorplan.

    Args:
        cmap (str): name of the color map to use when rendering the image.
          See: https://matplotlib.org/stable/users/explain/colors/colormaps.html
        show (bool): whether or not to show the image.
        save (bool): whether or not to save the image (as a .png file).
        image_filepath (str): a custom filepath to use when saving the image.
          only applies if `save=True`.
    """
    plt.imshow(self.floorplan, interpolation="nearest", cmap=cmap)
    if show:
      plt.show()
    if save:
      image_filepath = image_filepath or self.floorplan_image_filepath
      plt.savefig(image_filepath)

  @property
  def device_layout_map_filepath(self):
    return os.path.join(self.tabular_dirpath, "device_layout_map.json")

  @cached_property
  def device_layout_map(self):
    """The building's device layout map."""
    with open(self.device_layout_map_filepath, encoding="utf-8") as json_file:
      return json.load(json_file)

  @property
  def device_infos_filepath(self):
    return os.path.join(self.tabular_dirpath, "device_info_dicts.pickle")

  @cached_property
  def device_infos(self):
    return pickle.load(open(self.device_infos_filepath, "rb"))

  @property
  def zone_infos_filepath(self):
    return os.path.join(self.tabular_dirpath, "zone_info_dicts.pickle")

  @cached_property
  def zone_infos(self):
    return pickle.load(open(self.zone_infos_filepath, "rb"))


class BuildingDatasetPartition(BuildingDataset):
  """A helper class for handling a specific dataset partition.

  Args:
    building_id (str): The identifier of the building (e.g. "sb1").
    partition_id (str): The identifier of a dataset partition (e.g. "2022_a").
    download (bool): Whether or not to download the dataset.
  """

  def __init__(self, partition_id, building_id="sb1", download=True):
    super().__init__(building_id=building_id, download=download)
    self.partition_id = partition_id

    if self.partition_id not in self.partition_ids:
      raise ValueError(f"Invalid partition: {self.partition_id}.")

  @property
  def partition_dirpath(self):
    return os.path.join(self.tabular_dirpath, self.building_id, self.partition_id)  # pylint:disable=line-too-long

  @property
  def data_filepath(self):
    return os.path.join(self.partition_dirpath, "data.npy.npz")

  @cached_property
  def data(self):
    return np.load(self.data_filepath)

  @property
  def metadata_filepath(self):
    return os.path.join(self.partition_dirpath, "metadata.pickle")

  @cached_property
  def metadata(self):
    metadata = pickle.load(open(self.metadata_filepath, "rb"))

    if "device_infos" not in metadata.keys():
      metadata["device_infos"] = self.device_infos

    if "zone_infos" not in metadata.keys():
      metadata["zone_infos"] = self.zone_infos

    return metadata


if __name__ == "__main__":

  ds = BuildingDataset()
  ds.display_floorplan(show=False, save=True)
