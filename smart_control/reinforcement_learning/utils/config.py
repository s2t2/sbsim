"""Centralized configuration for reinforcement learning paths and Gin bindings.

This module defines common file system paths used throughout the reinforcement
learning part of the `smart_control` project. It also provides several
Gin-configurable functions that allow these paths and other configurations
(like specific data files or preconfigured objects) to be easily managed and
overridden through Gin configuration files.

The imports at the top of the file, though many are marked as `unused-import`
by pylint, are necessary for Gin's registration mechanism. They make various
classes and functions from the simulation, reward, and utility modules available
for configuration via Gin files without needing to explicitly import them elsewhere
when using Gin.
"""

import os
from typing import Any

import gin
import numpy as np

# pylint: disable=unused-import
# These imports are necessary for Gin's auto-registration mechanism.
# They make the classes and functions available for configuration in Gin files
# without requiring explicit imports in the Gin config itself or calling code.
from smart_control.reward.electricity_energy_cost import ElectricityEnergyCost
from smart_control.reward.natural_gas_energy_cost import NaturalGasEnergyCost
from smart_control.reward.setpoint_energy_carbon_regret import SetpointEnergyCarbonRegretFunction
from smart_control.simulator.air_handler import AirHandler
from smart_control.simulator.boiler import Boiler
from smart_control.simulator.building import MaterialProperties
from smart_control.simulator.hvac_floorplan_based import FloorPlanBasedHvac
from smart_control.simulator.randomized_arrival_departure_occupancy import RandomizedArrivalDepartureOccupancy
from smart_control.simulator.simulator_building import SimulatorBuilding
from smart_control.simulator.stochastic_convection_simulator import StochasticConvectionSimulator
from smart_control.simulator.tf_simulator import TFSimulator
from smart_control.simulator.weather_controller import ReplayWeatherController
from smart_control.utils import controller_reader
from smart_control.utils import histogram_reducer
from smart_control.utils.controller_writer import ProtoWriterFactory
from smart_control.utils.environment_utils import to_timestamp
from smart_control.utils.observation_normalizer import StandardScoreObservationNormalizer
# pylint: enable=unused-import

# --- Path Constants ---
# These constants define key directory locations for the project.

# ROOT_DIR: The absolute path to the root directory of the `smart_control` project.
# Assumes this config.py file is located at:
# smart_control/smart_control/reinforcement_learning/utils/config.py
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# DATA_PATH: Base directory for shared data resources, specifically for the
# "sb1" (Smart Building 1) configuration set.
DATA_PATH = os.path.join(ROOT_DIR, "smart_control", "configs", "resources", "sb1")

# CONFIG_PATH: Directory containing Gin configuration files for training and
# simulation, specifically for the "sb1" setup.
CONFIG_PATH = os.path.join(ROOT_DIR, "smart_control", "configs", "resources", "sb1", "train_sim_configs")

# METRICS_PATH: Default base directory for storing metrics generated during
# RL agent training or evaluation.
METRICS_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "experiment_results", "metrics")

# RENDERS_PATH: Default base directory for saving rendered images or
# visualizations of the environment state.
RENDERS_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "experiment_results", "renders")

# OUTPUT_DATA_PATH: Default base directory for storing other output data,
# such as pre-populated "starter" replay buffers.
OUTPUT_DATA_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "data", "starter_buffers")

# EXPERIMENT_RESULTS_PATH: Root directory where all outputs for a specific
# RL experiment (metrics, renders, saved models, TensorBoard logs) are stored.
EXPERIMENT_RESULTS_PATH = os.path.join(ROOT_DIR, "smart_control", "reinforcement_learning", "experiment_results")


@gin.configurable
def get_histogram_path() -> str:
  """Returns the base path for histogram-related data.

  This path typically points to the directory containing precomputed histograms
  or data used for generating them. It is Gin-configurable, so it can be
  overridden in Gin configuration files if histogram data is located elsewhere.

  Returns:
    A string representing the file system path to histogram data resources.
    By default, this is the same as `DATA_PATH`.
  """
  return DATA_PATH


@gin.configurable
def get_reset_temp_values() -> np.ndarray:
  """Loads and returns initial temperature values for environment reset.

  These values are typically used to set the initial thermal state of the
  building simulation when an RL environment is reset. The temperatures are
  loaded from a `.npy` file. This function is Gin-configurable.

  Returns:
    A NumPy array containing the reset temperature values.
  """
  reset_temps_filepath = os.path.join(DATA_PATH, "reset_temps.npy")
  return np.load(reset_temps_filepath)


@gin.configurable
def get_zone_path() -> str:
  """Returns the file path to the building's zone layout data.

  This data usually defines the spatial layout or mask of different zones within
  the building, often stored as a `.npy` file. It is Gin-configurable.

  Returns:
    A string representing the file system path to the zone layout data.
  """
  # Example: "double_resolution_zone_1_2.npy" might define zones for floors 1 and 2.
  return os.path.join(DATA_PATH, "double_resolution_zone_1_2.npy")


@gin.configurable
def get_metrics_path() -> str:
  """Returns the default base path for storing experiment metrics.

  This is Gin-configurable, allowing users to specify a different directory
  for metrics output via Gin configuration files.

  Returns:
    A string representing the file system path for saving metrics.
    By default, this is `METRICS_PATH/metrics`.
  """
  return os.path.join(METRICS_PATH, "metrics")


@gin.configurable
def get_weather_path() -> str:
  """Returns the file path to the weather data CSV file.

  This weather data is used by the building simulation to model external
  thermal conditions. The path is Gin-configurable.

  Returns:
    A string representing the file system path to the weather data file.
  """
  # Example: "local_weather_moffett_field_20230701_20231122.csv"
  return os.path.join(
      DATA_PATH, "local_weather_moffett_field_20230701_20231122.csv"
  )


@gin.configurable
def get_histogram_reducer() -> histogram_reducer.HistogramReducer:
  """Creates and returns a pre-configured `HistogramReducer` instance.

  This function initializes a `histogram_reducer.HistogramReducer` with a
  specific set of parameters for binning certain observation features (like
  zone air temperature, damper commands, and flowrate setpoints). It uses a
  `ProtoReader` to access necessary data from `DATA_PATH`.

  The entire configuration of this reducer, including bin edges and which
  features to reduce, can be overridden via Gin configuration.

  Returns:
    An instance of `histogram_reducer.HistogramReducer` configured with
    default parameters for common smart building observation features.
  """
  # Default histogram parameters:
  # Defines which features to bin and the edges for those bins.
  # ("feature_name", (bin_edge1, bin_edge2, ...))
  histogram_parameters_tuples = (
      ("zone_air_temperature_sensor", (
          285.0, 286.0, 287.0, 288.0, 289.0, 290.0, 291.0, 292.0, 293.0,
          294.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 301.0, 302.0, 303.0,
      )),
      ("supply_air_damper_percentage_command", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
      ("supply_air_flowrate_setpoint", (
          0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9
      )),
  )
  # Initialize a reader to potentially load data needed by the reducer (e.g., for normalization stats)
  reader = controller_reader.ProtoReader(DATA_PATH)

  hr = histogram_reducer.HistogramReducer(
      histogram_parameters_tuples=histogram_parameters_tuples,
      reader=reader, # Reader might be used for data-dependent normalization
      normalize_reduce=True, # Indicates if data should be normalized before reduction
  )
  return hr
