"""Utilities for reducing the dimensionality of observation spaces.

This module provides classes and functions for transforming high-dimensional
observation data into a lower-dimensional representation. This can be useful
for improving the efficiency and performance of reinforcement learning agents,
especially when dealing with complex environments like real buildings that may
have many sensors of the same type (e.g., numerous temperature sensors).

The primary mechanisms for reduction include:
-   Calculating statistical summaries (e.g., mean, median, std) for groups of
    similar features.
-   (Potentially, though not fully detailed here) Histogram-based reduction,
    where feature values are binned.

The module defines base classes `BaseReducer` and `BaseReducedSequence` to
establish an interface for different reduction strategies and their outputs.
"""

import abc
import collections
from typing import Any, Callable, Mapping, Sequence, TypeAlias # Added TypeAlias

import gin
import numpy as np
import pandas as pd

# Mapping from common statistical function names to their NumPy implementations.
_STATS_FUNC_MAPPING: Mapping[str, Callable[..., np.ndarray]] = {
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    # Add other functions like np.min, np.max, np.percentile as needed.
}

# Type alias for a feature name (typically a string).
FeatureName: TypeAlias = str


@gin.configurable
def name_to_func(
    func_names: Sequence[str]
) -> Sequence[Callable[..., float]]:
  """Converts a sequence of statistical function names to callable functions.

  This Gin-configurable function looks up NumPy functions based on string names
  (e.g., "mean", "std", "median") provided in `func_names`.

  Args:
    func_names (Sequence[str]): A sequence of strings, where each string is a
      key in `_STATS_FUNC_MAPPING` (e.g., ["mean", "std"]).

  Returns:
    Sequence[Callable[..., float]]: A sequence of callable NumPy functions
    corresponding to the input names.

  Raises:
    ValueError: If any name in `func_names` is not found in
      `_STATS_FUNC_MAPPING`.
  """
  callable_funcs: List[Callable[..., float]] = []
  for name_str in func_names:
    if name_str not in _STATS_FUNC_MAPPING:
      raise ValueError(
          f"Requested function '{name_str}' is not supported. "
          f"Available functions: {list(_STATS_FUNC_MAPPING.keys())}"
      )
    callable_funcs.append(_STATS_FUNC_MAPPING[name_str])
  return callable_funcs


class BaseReducedSequence(metaclass=abc.ABCMeta):
  """Abstract base class for a sequence of reduced observations.

  This class represents data that has undergone a dimensionality reduction
  process. It stores the `reduced_sequence` (typically a Pandas DataFrame)
  and defines an abstract `expand` method that implementing classes must
  provide to attempt reconstruction of the original data dimensionality.

  Attributes:
    reduced_sequence (pd.DataFrame): The DataFrame containing the
      dimensionality-reduced observation data.
  """
  reduced_sequence: pd.DataFrame

  @abc.abstractmethod
  def expand(self) -> pd.DataFrame:
    """Reconstructs the data to its original dimensionality (potentially lossy).

    Implementing classes should define how the `reduced_sequence` can be
    transformed back into a format that approximates the original,
    higher-dimensional observation space.

    Returns:
      pd.DataFrame: A DataFrame representing the expanded (reconstructed) data.
    """


class BaseReducer(metaclass=abc.ABCMeta):
  """Abstract base class for dimensionality reduction strategies.

  Defines the interface for classes that can reduce the dimensionality of an
  observation sequence (e.g., by calculating statistics, applying PCA, etc.).
  """

  @abc.abstractmethod
  def reduce(self, observation_sequence: pd.DataFrame) -> BaseReducedSequence:
    """Transforms a raw observation sequence into a reduced representation.

    Args:
      observation_sequence (pd.DataFrame): A DataFrame where rows are time
        steps and columns represent different features or sensor readings from
        the environment.

    Returns:
      BaseReducedSequence: An object containing the reduced observation
      sequence and capable of (approximately) expanding it back.
    """


@gin.configurable
class IdentityReducer(BaseReducer):
  """A pass-through reducer that performs no dimensionality reduction.

  This class implements the `BaseReducer` interface but does not change the
  input `observation_sequence`. It's useful for compatibility in pipelines
  where a reducer is expected but no reduction is desired, or as a baseline.
  """

  class IdentityReducedSequence(BaseReducedSequence):
    """Represents an un-reduced sequence for the `IdentityReducer`."""
    def expand(self) -> pd.DataFrame:
      """Returns the original (un-reduced) sequence.

      Returns:
        pd.DataFrame: The `reduced_sequence` itself, as no transformation
        was applied.
      """
      return self.reduced_sequence

  def reduce(self, observation_sequence: pd.DataFrame) -> BaseReducedSequence:
    """Returns the input observation sequence without modification.

    Args:
      observation_sequence (pd.DataFrame): The input observation data.

    Returns:
      IdentityReducedSequence: An object containing the original, un-reduced
      `observation_sequence`.
    """
    rs = self.IdentityReducedSequence()
    rs.reduced_sequence = observation_sequence
    return rs


@gin.configurable
class StatsReducer(BaseReducer):
  """Reduces features by calculating statistical summaries (e.g., mean, median).

  This reducer groups similar features (e.g., all zone temperature sensors)
  and calculates one or more statistical values (like mean, std, median) for
  each group at each time step. Features listed in `passthrough_features` are
  not reduced and are carried over to the output.

  The `expand` method reconstructs an approximate full observation sequence by
  assigning the calculated statistic (typically the first one in `stats_funcs`,
  e.g., mean) back to all original features within that group.
  """

  class StatsReducedSequence(BaseReducedSequence):
    """Stores statistically reduced data and supports approximate expansion.

    Attributes:
      reduced_sequence (pd.DataFrame): DataFrame where original feature columns
        are replaced by columns of their calculated statistics (e.g.,
        ('zone_air_temperature_sensor', 'mean')). Passthrough features are
        also included.
      _passthrough_features (Sequence[Any]): List of feature names that were
        not reduced.
      _stats_funcs (Sequence[Callable[..., float]]): Statistical functions used
        for reduction (e.g., [np.mean, np.std]).
      _feature_mapping (Mapping[FeatureName, Sequence[Any]]): Maps generic
        feature names to the list of original column names that were grouped
        under it for statistical calculation.
    """
    def __init__(
        self,
        passthrough_features: Sequence[Any], # Column names (str or tuple)
        stats_funcs: Sequence[Callable[..., float]],
        feature_mapping: Mapping[FeatureName, Sequence[Any]], # Original column names
    ):
      self._passthrough_features = passthrough_features
      self._stats_funcs = stats_funcs
      self._feature_mapping = feature_mapping
      # self.reduced_sequence is set by the StatsReducer.reduce method

    def expand(self) -> pd.DataFrame:
      """Reconstructs an approximate full observation DataFrame.

      For each original feature that was reduced, its value at each time step
      is approximated by the primary statistic (the first function in
      `_stats_funcs`) calculated for its group. Passthrough features are
      restored directly.

      Returns:
        pd.DataFrame: An expanded DataFrame where reduced features are
        approximated. The index matches `self.reduced_sequence`.
      """
      # Initialize a dictionary to build the expanded DataFrame
      expanded_data: Dict[Any, pd.Series] = {}

      # Add passthrough features directly
      for feature_col_name in self._passthrough_features:
        if feature_col_name in self.reduced_sequence.columns:
          expanded_data[feature_col_name] = self.reduced_sequence[feature_col_name]

      # Reconstruct reduced features
      # The primary statistic (first in _stats_funcs) is used for reconstruction.
      primary_stat_func_name = self._stats_funcs[0].__name__ if self._stats_funcs else "mean"

      for generic_feature_name, original_cols_for_feature in self._feature_mapping.items():
        # Column name for the primary statistic in the reduced DataFrame
        stat_col_name = (generic_feature_name, primary_stat_func_name)
        if stat_col_name in self.reduced_sequence.columns:
          # Assign this statistic's time series to all original columns
          # that were part of this feature group.
          for original_col_name in original_cols_for_feature:
            expanded_data[original_col_name] = self.reduced_sequence[stat_col_name]
        else:
          logging.warning(
              "Statistic column '%s' not found in reduced sequence for feature '%s'. "
              "Cannot expand this feature.",
              stat_col_name, generic_feature_name
          )
      return pd.DataFrame(expanded_data, index=self.reduced_sequence.index)

  def __init__(
      self,
      passthrough_features: Sequence[Any], # Column names (str or tuple)
      stats_funcs_names: Sequence[str], # e.g., ["mean", "std"]
  ):
    """Initializes the StatsReducer.

    Args:
      passthrough_features (Sequence[Any]): A sequence of column names
        (can be strings or tuples for MultiIndex) from the input DataFrame that
        should be passed through without any reduction.
      stats_funcs_names (Sequence[str]): A sequence of strings representing the
        names of statistical functions to apply (e.g., "mean", "median", "std").
        These names must be keys in `_STATS_FUNC_MAPPING`.

    Raises:
      ValueError: If `stats_funcs_names` is empty.
    """
    self._passthrough_features: Sequence[Any] = passthrough_features
    self._stats_funcs: Sequence[Callable[..., float]] = name_to_func(stats_funcs_names)

    if not self._stats_funcs:
      raise ValueError("At least one statistical function must be provided.")

  def _get_feature_to_columns_mapping(
      self, observation_sequence: pd.DataFrame
  ) -> Mapping[FeatureName, Sequence[Any]]: # Value is list of original column names
    """Groups observation columns by their generic feature type.

    For DataFrames with MultiIndex columns (e.g., (device_id, feature_name)),
    this method groups column names by the last element of the tuple (assumed
    to be the generic feature name like "zone_air_temperature_sensor").
    It excludes columns listed in `_passthrough_features`.

    Args:
      observation_sequence (pd.DataFrame): The input DataFrame.

    Returns:
      Mapping[FeatureName, Sequence[Any]]: A dictionary where keys are
      generic feature names (str) and values are lists of the original column
      names (which can be str or tuple) that belong to that feature group.
    """
    feature_map = collections.defaultdict(list)
    for original_col_name in observation_sequence.columns:
      if original_col_name in self._passthrough_features:
        continue

      generic_feature_name: FeatureName
      if isinstance(original_col_name, tuple) and original_col_name:
        generic_feature_name = str(original_col_name[-1]) # Assume last element is feature type
      else:
        generic_feature_name = str(original_col_name)
      feature_map[generic_feature_name].append(original_col_name)
    return feature_map

  def reduce(self, observation_sequence: pd.DataFrame) -> BaseReducedSequence:
    """Reduces the observation sequence using specified statistical functions.

    Args:
      observation_sequence (pd.DataFrame): Input DataFrame of observations.

    Returns:
      StatsReducedSequence: An object containing the reduced DataFrame.
        The reduced DataFrame will have columns for passthrough features and
        new columns for each statistic calculated per feature group (e.g.,
        ('zone_air_temperature_sensor', 'mean')).
    """
    feature_group_to_original_cols_map = self._get_feature_to_columns_mapping(
        observation_sequence
    )

    # Start with passthrough features
    reduced_parts: List[pd.DataFrame] = []
    if self._passthrough_features:
        # Ensure only existing columns are selected
        existing_passthrough = [
            col for col in self._passthrough_features
            if col in observation_sequence.columns
        ]
        if existing_passthrough:
            reduced_parts.append(observation_sequence[existing_passthrough])


    # Calculate stats for each feature group
    for generic_feature_name, original_cols in feature_group_to_original_cols_map.items():
      if not original_cols: # Should not happen if map built correctly
          continue
      # Extract the subset of data for this feature group
      feature_subset_df = observation_sequence[list(original_cols)]
      stats_for_feature: Dict[Tuple[FeatureName, str], pd.Series] = {}
      for stat_func in self._stats_funcs:
        # Apply stat_func row-wise across columns of this feature group
        # New column name will be (generic_feature_name, stat_func_name)
        new_col_name = (generic_feature_name, stat_func.__name__)
        stats_for_feature[new_col_name] = stat_func(feature_subset_df, axis=1)
      reduced_parts.append(pd.DataFrame(stats_for_feature, index=observation_sequence.index))

    # Concatenate all parts to form the reduced sequence
    if not reduced_parts: # If no features to reduce and no passthrough
        final_reduced_df = pd.DataFrame(index=observation_sequence.index)
    else:
        final_reduced_df = pd.concat(reduced_parts, axis=1)

    # Create and return the StatsReducedSequence object
    reduced_seq_obj = self.StatsReducedSequence(
        passthrough_features=self._passthrough_features,
        stats_funcs=self._stats_funcs,
        feature_mapping=feature_group_to_original_cols_map,
    )
    reduced_seq_obj.reduced_sequence = final_reduced_df
    return reduced_seq_obj
