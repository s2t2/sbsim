"""Smart Buildings Dataset implementation, including loading and downloading."""

from functools import cached_property
import json
import os
import pickle
import shutil

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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
    ```python
    ds = BuildingDataset(building_id="sb1", download=True)
    ```
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
    """The name of the local zip file after it is has been downloaded."""
    return f"{self.building_id}.zip"

  @property
  def zip_url(self):
    """The URL of the zip file located on Google Cloud Storage."""
    return (
        "https://storage.googleapis.com/gresearch/smart_buildings_dataset/"
        f"tabular_data/{self.zip_filename}"
    )

  @property
  def zip_filepath(self):
    """The filepath of the local zip file after it has been downloaded."""
    return os.path.join(DATA_DIR, self.zip_filename)

  @property
  def building_dirpath(self):
    """The local directory containing the building's dataset, after it has been
    extracted from the local zip file.
    """
    return os.path.join(DATA_DIR, self.building_id)

  def download(self, timeout=60):
    """Downloads the building's dataset from Google Cloud Storage.

    Only downloads and unzips the dataset if it doesn't already exist at the
      expected [`building_dirpath`](./#smart_control.dataset.dataset.BuildingDataset.building_dirpath)
      location.

    Download speed is fairly quick, but unzipping takes a few moments.
    """
    if os.path.isdir(self.building_dirpath):
      print("Using previously-downloaded data...")
      print(os.path.abspath(self.building_dirpath))
    else:
      print("Downloading zip file...")
      print(self.zip_url)
      with requests.get(self.zip_url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(self.zip_filepath, "wb") as f:
          for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

      print("Unpacking zip file...")
      print(os.path.abspath(self.zip_filepath))
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
    """Filepath for saving an image of the floorplan."""
    floorplan_image_filename = f"{self.building_id}_floorplan.png"
    return os.path.join(self.building_dirpath, floorplan_image_filename)

  def display_floorplan(
      self,
      cmap="binary",
      show=True,
      save=True,
      image_filepath: str | None = None,
  ):
    """Renders an image of the building's floorplan.

    Here is an example floorplan for building "sb1":

    ![An image of a floorplan.](../assets/images/sb1_floorplan.png)

    Args:
      cmap (str): The name of a [matplotlib color map](https://matplotlib.org/stable/users/explain/colors/colormaps.html)
        to use when rendering the image.
      show (bool): Whether or not to show the image.
      save (bool): Whether or not to save the image (as a .png file).
      image_filepath (str): An optional custom filepath to use when saving the
        image. Only applies if `save=True`. By default, saves to the [`floorplan_image_filepath`](./#smart_control.dataset.dataset.BuildingDataset.floorplan_image_filepath)
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

  @cached_property
  def devices_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A dataframe containing information about the building's devices.

    Each row is uniquely identified by the "device_id".

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        |    |          device_id | namespace   | code              |   device_type | observable_fields                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | action_fields                                                                                                                                                                                                                                                                                                                                                                                                       |
        |---:|-------------------:|:------------|:------------------|--------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        |  0 | 202194278473007104 | PHRED       | SB1:AHU:AC-2      |             6 | {'building_air_static_pressure_sensor': 1, 'outside_air_flowrate_sensor': 1, 'supply_fan_speed_percentage_command': 1, 'supply_air_temperature_sensor': 1, 'supply_fan_speed_frequency_sensor': 1, 'supply_air_static_pressure_setpoint': 1, 'return_air_temperature_sensor': 1, 'mixed_air_temperature_setpoint': 1, 'exhaust_fan_speed_percentage_command': 1, 'exhaust_fan_speed_frequency_sensor': 1, 'outside_air_damper_percentage_command': 1, 'mixed_air_temperature_sensor': 1, 'exhaust_air_damper_percentage_command': 1, 'cooling_percentage_command': 1, 'outside_air_flowrate_setpoint': 1, 'supply_air_temperature_setpoint': 1, 'building_air_static_pressure_setpoint': 1, 'supply_air_static_pressure_sensor': 1} | {'exhaust_air_damper_percentage_command': 1, 'supply_air_temperature_setpoint': 1, 'supply_fan_speed_percentage_command': 1, 'outside_air_flowrate_setpoint': 1, 'cooling_percentage_command': 1, 'mixed_air_temperature_setpoint': 1, 'exhaust_fan_speed_percentage_command': 1, 'outside_air_damper_percentage_command': 1, 'supply_air_static_pressure_setpoint': 1, 'building_air_static_pressure_setpoint': 1} |
        |  1 |   2760348383893915 | CDM         | VAV CO 1-1-10 CO2 |             4 | {'zone_air_heating_temperature_setpoint': 1, 'zone_air_temperature_sensor': 1, 'zone_air_co2_concentration_sensor': 1, 'supply_air_flowrate_setpoint': 1, 'zone_air_co2_concentration_setpoint': 1, 'zone_air_cooling_temperature_setpoint': 1, 'supply_air_flowrate_sensor': 1, 'supply_air_damper_percentage_command': 1}                                                                                                                                                                                                                                                                                                                                                                                                         | {'supply_air_damper_percentage_command': 1, 'supply_air_flowrate_setpoint': 1, 'zone_air_heating_temperature_setpoint': 1, 'zone_air_cooling_temperature_setpoint': 1, 'zone_air_co2_concentration_setpoint': 1}                                                                                                                                                                                                    |
        |  2 |   2562701969438717 | CDM         | VAV CO 2-2-36     |             4 | {'zone_air_heating_temperature_setpoint': 1, 'supply_air_flowrate_sensor': 1, 'supply_air_flowrate_setpoint': 1, 'supply_air_damper_percentage_command': 1, 'zone_air_temperature_sensor': 1, 'zone_air_cooling_temperature_setpoint': 1}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | {'supply_air_flowrate_setpoint': 1, 'supply_air_damper_percentage_command': 1, 'zone_air_cooling_temperature_setpoint': 1, 'zone_air_heating_temperature_setpoint': 1}                                                                                                                                                                                                                                              |
        |  3 |   2806035809406684 | CDM         |                   |             4 | {'discharge_air_temperature_setpoint': 1, 'supply_air_flowrate_sensor': 1, 'zone_air_heating_temperature_setpoint': 1, 'heating_water_valve_percentage_command': 1, 'supply_air_damper_percentage_command': 1, 'supply_air_flowrate_setpoint': 1, 'discharge_air_temperature_sensor': 1, 'zone_air_cooling_temperature_setpoint': 1, 'zone_air_temperature_sensor': 1}                                                                                                                                                                                                                                                                                                                                                              | {'discharge_air_temperature_setpoint': 1, 'heating_water_valve_percentage_command': 1, 'zone_air_cooling_temperature_setpoint': 1, 'supply_air_damper_percentage_command': 1, 'zone_air_heating_temperature_setpoint': 1, 'supply_air_flowrate_setpoint': 1}                                                                                                                                                        |
        |  4 |   2790439929052995 | CDM         | VAV CO 1-1-43     |             4 | {'zone_air_heating_temperature_setpoint': 1, 'supply_air_flowrate_setpoint': 1, 'zone_air_temperature_sensor': 1, 'zone_air_cooling_temperature_setpoint': 1, 'supply_air_flowrate_sensor': 1, 'supply_air_damper_percentage_command': 1}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | {'supply_air_damper_percentage_command': 1, 'zone_air_heating_temperature_setpoint': 1, 'supply_air_flowrate_setpoint': 1, 'zone_air_cooling_temperature_setpoint': 1}                                                                                                                                                                                                                                              |
    """
    # pylint: enable=line-too-long
    df = pd.DataFrame(self.device_infos)
    df = df.drop(columns=["zone_id"])  # many to many relationship
    return df

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

  @cached_property
  def zones_df(self) -> pd.DataFrame:
    # pylint: disable=line-too-long
    """A dataframe containing information about the building's zones.

    Each row is uniquely identified by the "zone_id".

    Returns:
      A `pandas.DataFrame`. Here is an example of the structure:

        |    | zone_id             | building_id          | zone_description   |   area |   zone_type |   floor | devices                                  |   n_devices |
        |---:|:--------------------|:---------------------|:-------------------|-------:|------------:|--------:|:-----------------------------------------|------------:|
        |  0 | rooms/1002000133978 | buildings/3616672508 | SB1-2-C2054        |      0 |           1 |       2 | ['2618581107144046', '2696593986887004'] |           2 |
        |  1 | rooms/9028471695    | buildings/3616672508 | SB1-2-2D4A         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
        |  2 | rooms/9028472496    | buildings/3616672508 | SB1-2-2D4H         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
        |  3 | rooms/9028558963    | buildings/3616672508 | SB1-2-2D4B         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
        |  4 | rooms/9028483453    | buildings/3616672508 | SB1-2-2D4G         |      0 |           1 |       2 | ['2696593986887004']                     |           1 |
    """
    # pylint: enable=line-too-long
    df = pd.DataFrame(self.zone_infos)
    df["n_devices"] = df["devices"].apply(len)
    return df


class BuildingDatasetPartition:
  """A helper class for handling a specific dataset partition.

  Args:
    building_dataset (BuildingDataset): The building dataset.
    partition_id (str): The identifier of a partition in the specified dataset
      (e.g. "2022_a").

  Example:
    ```python
    ds = BuildingDataset(building_id='sb1', download=True)
    partition = BuildingDatasetPartition(
       building_dataset=ds, partition_id='2022_a'
    )
    ```
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
