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

# import pandas as pd


DATA_DIR = os.path.join(ROOT_DIR, "data")

VALID_BUILDING_PARTITIONS = {
    "sb1": ["2022_a", "2022_b", "2023_a", "2023_b", "2024_a"]
}


class BuildingDataset:
  """A helper class for handling the dataset for a specific building.

  Args:
    building_id (str): The identifier of the building (e.g. "sb1").
    download (bool): Whether or not to download the dataset.

  Examples:
    >>> ds = BuildingDataset(building_id="sb1", download=True)
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
  def floorplan(self) -> np.ndarray:
    """The building's floorplan, as a numpy array.

    The floorplan consists of a map of pixels. Here is a mapping of the values:

      + 0: inside / internal space
      + 1: wall / boundary
      + 2: outside / external space

    Use the [`display_floorplan`](./#smart_control.dataset.dataset.BuildingDataset.display_floorplan)
      method to view an image of the floorplan.
    """
    return np.load(self.floorplan_filepath)

  @property
  def floorplan_image_filepath(self):
    floorplan_image_filename = f"{self.building_id}_floorplan.png"
    return os.path.join(self.building_dirpath, floorplan_image_filename)

  def display_floorplan(
      self, cmap="binary", show=True, save=True, image_filepath=None
  ):
    """Renders an image of the building's floorplan.

    Here is an example floorplan for building "sb1":

    ![An image of a floorplan.](../assets/images/sb1_floorplan.png)

    Args:
      cmap (str): The name of a [matplotlib color map](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        to use when rendering the image.
      show (bool): Whether or not to show the image.
      save (bool): Whether or not to save the image (as a .png file).
      image_filepath (str): A custom filepath to use when saving the image.
        Only applies if `save=True`.
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
  def device_layout_map(self) -> dict:
    """A layout map of devices in the building.

    Returns:
      A dictionary with keys corresponding to each of the devices
        (e.g. 'VAV CO 1-1-06'). Each value is a list of integer coordinates.
        The length of the coordinates list is not the same across all devices.
        Here is an abbreviated version of the device layout map:

        ```py
        {
          'VAV CO 1-1-06': [[79, 35], [80, 35], [80, 34], [80, 33], ... ],
          'VAV CO 1-1-07 CO2: [[93, 73], [94, 73], [94, 72], [94, 71], ... ],
          ...
          'VAV RH 1-1-28 CO2 (Tech Talk 1H2)': [[22, 422], [23, 422], ... ],
          ...
          'VAV RH 1-1-55': [[145, 126], [146, 126], [147, 126], ... ]
        }
        ```
    """
    with open(self.device_layout_map_filepath, encoding="utf-8") as json_file:
      return json.load(json_file)

  @property
  def device_infos_filepath(self):
    return os.path.join(self.tabular_dirpath, "device_info_dicts.pickle")

  @cached_property
  def device_infos(self) -> list[dict]:
    """Information about the devices in the building.

    Returns:
      A list of device dictionaries. Here is an example device dictionary:

        ```py
        {
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
        ```
    """
    return pickle.load(open(self.device_infos_filepath, "rb"))

  @property
  def zone_infos_filepath(self):
    return os.path.join(self.tabular_dirpath, "zone_info_dicts.pickle")

  @cached_property
  def zone_infos(self) -> list[dict]:
    """Information about the zones in the building.

    Returns:
      A list of zone dictionaries. Here is an example zone dictionary:

        ```py
        {
          'zone_id': 'rooms/1002000133978',
          'building_id': 'buildings/3616672508',
          'zone_description': 'SB1-2-C2054',
          'area': 0.0,
          'zone_type': 1,
          'floor': 2,
          'devices': ['2618581107144046', '2696593986887004'],
        }
        ```
    """
    return pickle.load(open(self.zone_infos_filepath, "rb"))

  # @cached_property
  # def zones_df(self) -> pd.DataFrame
  #  df = pd.DataFrame(ds.zone_infos)
  #  df["n_devices"] = df["devices"].apply(lambda x: len(x))
  #  return df


class BuildingDatasetPartition:
  """A helper class for handling a specific dataset partition.

  Args:
    building_dataset (BuildingDataset): The building dataset.
    partition_id (str): The identifier of a partition in the specified dataset
      (e.g. "2022_a").

  Example:
    >>> ds = BuildingDataset(building_id='sb1', download=True)
    >>> partition = BuildingDatasetPartition(
    >>>    building_dataset=ds, partition_id='2022_a'
    >>> )
  """

  def __init__(self, building_dataset: BuildingDataset, partition_id: str):
    self.ds = building_dataset
    self.partition_id = partition_id

    if self.partition_id not in self.ds.partition_ids:
      raise ValueError(f"Invalid partition: {self.partition_id}.")

  @property
  def partition_dirpath(self):
    return os.path.join(self.ds.tabular_dirpath, self.ds.building_id, self.partition_id)  # pylint:disable=line-too-long

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
      metadata["device_infos"] = self.ds.device_infos

    if "zone_infos" not in metadata.keys():
      metadata["zone_infos"] = self.ds.zone_infos

    return metadata


if __name__ == "__main__":

  ds = BuildingDataset()
  ds.display_floorplan(show=False, save=True)
