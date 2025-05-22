"""Utility functions for creating and configuring RL environments.

This module provides helper functions to simplify the instantiation and setup
of `smart_control.environment.Environment` instances, primarily by leveraging
Gin configuration files for defining the environment's parameters and components.
"""

from typing import Optional # For Optional type hint

import gin

from smart_control.environment.environment import Environment
from smart_control.reinforcement_learning.utils.constants import DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT


def load_environment(gin_config_file: str) -> Environment:
  """Loads and returns an `Environment` instance from a Gin configuration file.

  This function first clears any existing global Gin configurations to ensure a
  clean setup from the specified file. It then parses the Gin file, which
  should define all necessary bindings for instantiating the
  `smart_control.environment.Environment` along with its components (e.g.,
  building model, reward function, normalizers).

  Args:
    gin_config_file: The file system path to the Gin configuration file
      (e.g., "path/to/your/sim_config.gin").

  Returns:
    An instance of `smart_control.environment.Environment` configured
    according to the provided Gin file.
  """
  # Gin configurations are global. It's crucial to manage the config state,
  # especially when loading different configurations sequentially.
  # `gin.unlock_config()` allows modification of the global config.
  # `gin.clear_config()` removes all previous bindings.
  with gin.unlock_config():
    gin.clear_config()
    gin.parse_config_file(gin_config_file)
    # `Environment()` will now be constructed using bindings from `gin_config_file`.
    # Pylint disable is because Gin magically fills parameters for Environment.
    return Environment()  # pylint: disable=no-value-for-parameter


def create_and_setup_environment(
    gin_config_file: str,
    metrics_path: Optional[str] = None,
    occupancy_normalization_constant: float = DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT,
) -> Environment:
  """Creates an environment from a Gin file and applies additional setup.

  This function first loads an `Environment` instance using `load_environment`
  with the specified Gin configuration file. It then allows for programmatically
  setting or overriding certain environment attributes, such as the path for
  saving metrics and the constant used for normalizing occupancy data.

  Args:
    gin_config_file: The file system path to the Gin configuration file.
    metrics_path: An optional string specifying the directory path where
      environment metrics (e.g., observations, rewards) should be written.
      If `None`, metrics writing might be disabled or use a default path
      defined within the Environment or Gin config. Defaults to `None`.
    occupancy_normalization_constant: A float value used for normalizing
      occupancy signals within the environment. Defaults to
      `DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT`.

  Returns:
    A configured instance of `smart_control.environment.Environment`.

  Example:
    ```python
    from smart_control.reinforcement_learning.utils import environment

    gin_file = "path/to/my_sim_config.gin"
    exp_metrics_path = "/tmp/my_experiment_metrics"

    env = environment.create_and_setup_environment(
        gin_config_file=gin_file,
        metrics_path=exp_metrics_path,
        occupancy_normalization_constant=100.0
    )
    print(f"Environment configured with metrics path: {env.metrics_path}")
    ```
  """
  # Load the base environment configuration from the Gin file
  env = load_environment(gin_config_file)

  # Apply specific overrides or additional setup
  if metrics_path is not None: # Only override if a path is provided
    env.metrics_path = metrics_path
  env.occupancy_normalization_constant = occupancy_normalization_constant

  return env
