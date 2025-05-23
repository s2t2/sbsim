"""Configuration settings and Gin-configurable functions for RL.

This module defines paths to various data and configuration files used
throughout the reinforcement learning setup. It also provides Gin-configurable
functions to access these resources, allowing for flexible parameterization
of experiments.

The module imports several classes from the broader `smart_control` project.
These imports are primarily for Gin: by importing them here, they become
available for configuration via Gin files without needing to be explicitly
referenced in the code that uses this config module. This is a common pattern
for making classes and functions configurable with Gin.

Attributes:
  ROOT_DIR (str): Absolute path to the root directory of the project.
  DATA_PATH (str): Path to general data resources, typically for building 'sb1'.
  CONFIG_PATH (str): Path to Gin configuration files for training and simulation.
  METRICS_PATH (str): Default path for saving metrics during experiments.
  RENDERS_PATH (str): Default path for saving rendered visualizations.
  OUTPUT_DATA_PATH (str): Path for saving larger output data, like starter
    replay buffers.
  EXPERIMENT_RESULTS_PATH (str): Root path for storing all experiment results,
    including metrics, renders, and saved models.
"""

import os
from typing import Any

import gin
import numpy as np

# These imports make the respective classes/functions configurable by Gin
# when this module is imported, even if they are not directly used here.
# This is a common Gin pattern to register components.
from smart_control.reward import electricity_energy_cost # pylint: disable=unused-import
from smart_control.reward import natural_gas_energy_cost # pylint: disable=unused-import
from smart_control.reward import setpoint_energy_carbon_regret # pylint: disable=unused-import
from smart_control.simulator import air_handler # pylint: disable=unused-import
from smart_control.simulator import boiler # pylint: disable=unused-import
from smart_control.simulator import building as sim_building # pylint: disable=unused-import
from smart_control.simulator import hvac_floorplan_based # pylint: disable=unused-import
from smart_control.simulator import randomized_arrival_departure_occupancy # pylint: disable=unused-import
from smart_control.simulator import simulator_building as sim_bldg # pylint: disable=unused-import
from smart_control.simulator import stochastic_convection_simulator # pylint: disable=unused-import
from smart_control.simulator import tf_simulator # pylint: disable=unused-import
from smart_control.simulator import weather_controller # pylint: disable=unused-import
from smart_control.utils import controller_reader # pylint: disable=unused-import
from smart_control.utils import histogram_reducer as hist_reducer # pylint: disable=unused-import
from smart_control.utils import controller_writer # pylint: disable=unused-import
from smart_control.utils import environment_utils # pylint: disable=unused-import
from smart_control.utils import observation_normalizer # pylint: disable=unused-import

# Define absolute paths relative to this file's location.
# Assumes this file is at: .../smart_control/reinforcement_learning/utils/config.py
# So, ROOT_DIR goes up three levels.
_CURRENT_FILE_DIR = os.path.dirname(__file__)
ROOT_DIR: str = os.path.abspath(
    os.path.join(_CURRENT_FILE_DIR, "..", "..", "..")
)

# Data paths, primarily for the 'sb1' building scenario
DATA_PATH: str = os.path.join(
    ROOT_DIR, "smart_control", "configs", "resources", "sb1"
)
CONFIG_PATH: str = os.path.join(DATA_PATH, "train_sim_configs")

# Paths for experiment outputs
_RL_EXPERIMENT_ROOT = os.path.join(
    ROOT_DIR, "smart_control", "reinforcement_learning", "experiment_results"
)
METRICS_PATH: str = os.path.join(_RL_EXPERIMENT_ROOT, "metrics")
RENDERS_PATH: str = os.path.join(_RL_EXPERIMENT_ROOT, "renders")
OUTPUT_DATA_PATH: str = os.path.join(
    ROOT_DIR, "smart_control", "reinforcement_learning", "data", "starter_buffers"
)
EXPERIMENT_RESULTS_PATH: str = _RL_EXPERIMENT_ROOT


@gin.configurable
def get_histogram_path() -> str:
  """Returns the base path for histogram-related data.

  This path is configurable via Gin. By default, it points to `DATA_PATH`.

  Returns:
    str: The configured path to histogram data resources.
  """
  return DATA_PATH


@gin.configurable
def get_reset_temp_values() -> np.ndarray:
  """Loads and returns initial temperature values for environment reset.

  These values are loaded from a .npy file specified by `reset_temps.npy`
  within the `DATA_PATH`. The path can be implicitly reconfigured if
  `DATA_PATH` itself is changed via Gin, or this function could be made
  to accept a direct path.

  Returns:
    np.ndarray: A NumPy array containing temperature values used to reset
    the thermal state of the simulated building.
  """
  reset_temps_filepath = os.path.join(DATA_PATH, "reset_temps.npy")
  return np.load(reset_temps_filepath)


@gin.configurable
def get_zone_path() -> str:
  """Returns the path to the zone definition file.

  This file (e.g., `double_resolution_zone_1_2.npy`) likely contains data
  defining the layout or properties of thermal zones in the building.
  The path is relative to `DATA_PATH`.

  Returns:
    str: The path to the zone data file.
  """
  return os.path.join(DATA_PATH, "double_resolution_zone_1_2.npy")


@gin.configurable
def get_metrics_path() -> str:
  """Returns the base path for storing experiment metrics.

  This path is configurable via Gin. By default, it points to a 'metrics'
  subdirectory within `METRICS_PATH`.

  Returns:
    str: The configured root path for saving metrics.
  """
  return os.path.join(METRICS_PATH, "metrics") # Default sub-directory


@gin.configurable
def get_weather_path() -> str:
  """Returns the path to the weather data file.

  The weather file (e.g., a CSV) contains historical or typical meteorological
  year data used by the building simulation. The path is relative to
  `DATA_PATH`.

  Returns:
    str: The path to the weather data CSV file.
  """
  return os.path.join(
      DATA_PATH, "local_weather_moffett_field_20230701_20231122.csv"
  )


@gin.configurable
def get_histogram_reducer() -> hist_reducer.HistogramReducer:
  """Creates and returns a configured `HistogramReducer` instance.

  This function defines default parameters for histogram binning of certain
  observation features (zone air temperature, damper commands, flowrate
  setpoints). The `HistogramReducer` is used to transform continuous
  observation values into a discrete, binned representation, which can
  sometimes be beneficial for RL agent learning.

  The `DATA_PATH` is used to initialize a `ProtoReader` for the reducer,
  implying it might need to access other data from that location.

  Returns:
    hist_reducer.HistogramReducer: A configured instance of the
    `HistogramReducer`.
  """
  histogram_parameters_tuples = (
      (
          "zone_air_temperature_sensor",
          (
              285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0,
              294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0,
              303.0,
          ),
      ),
      (
          "supply_air_damper_percentage_command",
          (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
      ),
      (
          "supply_air_flowrate_setpoint",
          (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
      ),
  )
  # Assuming ProtoReader needs a base path to find various proto files if necessary
  reader = controller_reader.ProtoReader(DATA_PATH)

  hr = hist_reducer.HistogramReducer(
      histogram_parameters_tuples=histogram_parameters_tuples,
      reader=reader,
      normalize_reduce=True, # Normalizes data before binning
  )
  return hr
