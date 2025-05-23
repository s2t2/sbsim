"""Utility modules for the smart building control project.

This package consolidates a variety of helper functions and classes that
support different aspects of the smart building simulation, control, and
reinforcement learning pipeline. These utilities range from data conversion
and normalization to environment setup, data I/O, and visualization.

Modules within this package provide functionalities such as:
- `agent_utils`: Helpers related to reinforcement learning agents.
- `bounded_action_normalizer`: Normalization for actions with defined bounds.
- `building_image_generator`: Generating image representations of building states.
- `building_renderer`: Rendering building floor plans and thermal maps.
- `constants`: Shared constants used across the project.
- `controller_reader` & `controller_writer`: Reading and writing controller data.
- `conversion_utils`: Utilities for converting data types and units.
- `energy_utils`: Helpers for energy-related calculations.
- `environment_utils`: Utilities for setting up and managing simulation
  environments.
- `histogram_reducer`: Reducing time-series data into histograms.
- `observation_normalizer`: Normalizing observation data.
- `plot_utils`: Tools for creating various plots and visualizations.
- `reader_lib` & `writer_lib`: Base libraries for data reading and writing.
- `real_building_temperature_array_generator`: Generating temperature arrays
  from real building data.
- `reducer`: Base class for data reduction utilities.
- `regression_building_utils`: Utilities for regression-based building models.
- `run_command_predictor`: Predicting HVAC run commands.
- `visual_logger`: Logging visual data like images and plots.
"""
