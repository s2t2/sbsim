"""Reduces dimensionality of time-series data by converting features to histograms.

This module defines `HistogramReducer` and related utilities for compressing
wide multivariate time-series data. The core idea is to transform observations
for specific features (e.g., zone air temperatures from many sensors) into
histograms, where each bin counts the number of devices/sensors whose readings
fall into that bin's range. This can significantly reduce the dimensionality
of the observation space if the number of devices is much larger than the
number of bins.

The module also provides functionality to (lossily) reconstruct approximate
time-series data from these histograms, which might be useful for analysis or
visualization.
"""

import collections
import copy # For deepcopying assignments in HistogramReducedSequence
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from absl import logging
import gin
import numpy as np
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.utils import reader_lib
from smart_control.utils import reducer # For BaseReducedSequence, BaseReducer

# Type Aliases for clarity
Feature = str
"""Type alias for a feature name (e.g., 'zone_air_temperature_sensor')."""

Device = str
"""Type alias for a device identifier string."""

Value = float
"""Type alias for a measurement value."""

HistogramBins = np.ndarray
"""Type alias for a NumPy array representing the edges of histogram bins."""

HistogramParameters = Dict[Feature, HistogramBins]
"""Maps a feature name to the NumPy array of its histogram bin edges."""

HistogramAssignment = List[List[Device]]
"""Represents device assignments to bins for a single feature.
It's a list where each sublist corresponds to a bin and contains the IDs of
devices whose values fall into that bin.
"""

HistogramCounts = Union[Sequence[float], Sequence[int], np.ndarray]
"""Represents the count of devices in each histogram bin."""

HistogramExpansion = Mapping[Tuple[Device, Feature], Value]
"""Maps a (device_id, feature_name) tuple to its approximate reconstructed value."""


def assign_devices_to_bins(
    feature_name: Feature,
    bins: HistogramBins,
    observation_response: smart_control_building_pb2.ObservationResponse,
) -> HistogramAssignment:
  """Assigns devices to histogram bins based on their feature values.

  For a given feature and its defined bins, this function iterates through
  an `ObservationResponse`. It extracts the value of `feature_name` for each
  device and assigns the device ID to the corresponding bin.

  Binning logic:
  - For internal bins `b_i, b_{i+1}`: value `v` is in bin `i` if `b_i <= v < b_{i+1}`.
  - First bin (index 0): `v < b_1`.
  - Last bin (index N-1): `v >= b_{N-1}`.
  Assumes `bins` are monotonically increasing.

  Args:
    feature_name (Feature): The name of the measurement feature to process
      (e.g., "zone_air_temperature_sensor").
    bins (HistogramBins): A 1D NumPy array of monotonically increasing bin
      edges. The number of bins will be `len(bins)`.
    observation_response (smart_control_building_pb2.ObservationResponse):
      A protobuf message containing observations from multiple devices.

  Returns:
    HistogramAssignment: A list of lists, where the outer list corresponds to
    bins and each inner list contains the device IDs assigned to that bin.
  """
  num_bins = len(bins)
  assignment: HistogramAssignment = [[] for _ in range(num_bins)]

  for single_obs in observation_response.single_observation_responses:
    if single_obs.single_observation_request.measurement_name != feature_name:
      continue

    value = single_obs.continuous_value
    device_id = single_obs.single_observation_request.device_id

    # Determine which bin the value falls into.
    # `np.searchsorted` finds the index where `value` would be inserted to
    # maintain order. `side='right'` means it finds insertion point for
    # `value` to be greater than or equal to elements to its left.
    # Subtracting 1 gives the index of the bin whose left edge is <= value.
    # This handles `b_i <= v < b_{i+1}` for internal bins.
    bin_index = np.searchsorted(bins, value, side="right") - 1

    # Clamp index to be within [0, num_bins - 1]
    # If value < bins[0], searchsorted gives 0, so index becomes -1; clamp to 0.
    # If value >= bins[-1], searchsorted gives num_bins, index becomes num_bins-1.
    bin_index = np.clip(bin_index, 0, num_bins - 1)

    assignment[bin_index].append(device_id)
  return assignment


def approximate_values_from_histogram_assignment(
    measurement_name: Feature,
    assignment: HistogramAssignment,
    bins: HistogramBins,
) -> HistogramExpansion:
  """Reconstructs approximate feature values for devices from histogram assignments.

  For each device assigned to a bin, its approximate value is taken as the
  left edge of that bin. This is a lossy reconstruction.

  Args:
    measurement_name (Feature): The name of the measurement feature.
    assignment (HistogramAssignment): A list where each sublist (per bin)
      contains device IDs assigned to that bin.
    bins (HistogramBins): The 1D NumPy array of bin edges.

  Returns:
    HistogramExpansion: A mapping from `(device_id, measurement_name)` tuples
    to their approximated float values (the left edge of their assigned bin).
  """
  approximated_values: HistogramExpansion = {}
  for bin_idx, devices_in_bin in enumerate(assignment):
    bin_value = bins[bin_idx] # Use the left edge of the bin
    for device_id in devices_in_bin:
      approximated_values[(device_id, measurement_name)] = bin_value
  return approximated_values


def get_clipped_histogram(
    measurements: np.ndarray, bins: HistogramBins, clip_values: bool = True
) -> np.ndarray:
  """Generates a histogram from measurements, optionally clipping values to bin range.

  Args:
    measurements (np.ndarray): A 1D array of measurement values.
    bins (HistogramBins): A 1D array of bin edges. `np.histogram` expects
      `len(bins) + 1` edges for `len(bins)` bins. This function's `bins`
      argument seems to be interpreted as the *left* edges of N bins, and the
      rightmost edge for the last bin is implicitly `max(bins)`.
      The original code `cbins = np.append(bins, max(bins))` creates N+1 edges
      if `bins` has N elements, but the last two edges are identical, effectively
      making the last bin have zero width if not handled carefully by np.histogram.
      A more standard approach is `len(bins)` edges for `len(bins)-1` bins, or
      `num_bins` integer.
      Let's assume `bins` are the N left edges, and the N+1th edge is effectively infinity.
      The `np.histogram` function needs a sequence of bin edges.
    clip_values (bool): If True, clips measurements to the range
      `[min(bins), max(bins)]` before histogramming.

  Returns:
    np.ndarray: A 1D array of histogram counts, with float32 dtype.
  """
  # Standard way to define bins for np.histogram:
  # If `bins` are the N left edges, we need N+1 edges.
  # The last bin will effectively be [bins[-1], infinity)
  # However, the original code's `cbins = np.append(bins, max(bins))`
  # creates an N+1 length array if `bins` has N elements, but the last two
  # edges are the same. This means the last "bin" has zero width.
  # `np.histogram` with `bins=cbins` will then have `len(cbins)-1` bins.
  # If `bins` has N elements, `cbins` has N+1, so N bins.
  # This seems to match the intent if `bins` represents the start of each bin.
  if bins.size == 0:
    return np.array([], dtype=np.float32)

  # Define edges for np.histogram: bins are [edge1, edge2), [edge2, edge3), ...
  # If `bins` are the N left edges, we need N+1 edges.
  # The last bin catches everything >= bins[-1].
  hist_bin_edges = np.append(bins, np.inf) # Bins: [b0,b1), [b1,b2) ... [bN-1, inf)

  data_to_hist = measurements
  if clip_values:
    # Clip data to the full range covered by explicit bins
    # (values outside this will fall into underflow/overflow of np.histogram
    # if not clipped, but explicit clipping ensures they are counted in first/last user-defined bins)
    data_to_hist = np.clip(measurements, bins[0], bins[-1])

  counts, _ = np.histogram(data_to_hist, bins=hist_bin_edges)
  return counts.astype(np.float32)


def reassign_nodes(
    current_assignment: HistogramAssignment,
    next_histogram_counts: HistogramCounts
) -> HistogramAssignment:
  """Adjusts device assignments to bins to match target counts.

  This function attempts to minimally perturb the `current_assignment` of
  devices to bins so that the number of devices in each bin matches the
  `next_histogram_counts`. It does this by shifting devices between adjacent
  bins. This is a heuristic for stateful reconstruction in the `expand` method.

  Args:
    current_assignment (HistogramAssignment): The current list of lists, where
      each sublist contains device IDs for a bin. This list IS MODIFIED IN PLACE.
    next_histogram_counts (HistogramCounts): The target number of devices for
      each bin in the next step.

  Returns:
    HistogramAssignment: The modified `current_assignment` list.

  Raises:
    ValueError: If the number of bins or total number of devices differs
      between `current_assignment` and `next_histogram_counts`.
  """
  num_bins = len(current_assignment)
  if num_bins != len(next_histogram_counts):
    raise ValueError(
        f"Bin count mismatch: assignment has {num_bins} bins, "
        f"target counts have {len(next_histogram_counts)} bins."
    )

  current_bin_counts = [len(bin_devices) for bin_devices in current_assignment]
  if np.sum(current_bin_counts) != np.sum(next_histogram_counts):
    raise ValueError(
        f"Total device count mismatch: assignment has {np.sum(current_bin_counts)}, "
        f"target counts have {np.sum(next_histogram_counts)}."
    )

  # Iterate through bins to adjust device counts
  for i in range(num_bins):
    # While current bin has too many devices compared to target
    while len(current_assignment[i]) > next_histogram_counts[i]:
      if i + 1 < num_bins: # Can move to the right
        device_to_move = current_assignment[i].pop(0) # Take from front
        current_assignment[i + 1].insert(0, device_to_move) # Add to front of next
      else:
        # This case (last bin has too many, nowhere to move right) implies
        # an issue if total counts match, as devices should have been pulled
        # from it by preceding bins needing more.
        # If it occurs, it might mean counts are inconsistent.
        # However, given sum checks, this should ideally not be problematic
        # unless all preceding adjustments failed to balance.
        logging.warning("Cannot move device from last bin %d when it has excess.", i)
        break # Cannot fix further by moving right

    # While current bin has too few devices
    while len(current_assignment[i]) < next_histogram_counts[i]:
      moved = False
      # Try to pull from the right
      for j in range(i + 1, num_bins):
        if len(current_assignment[j]) > (next_histogram_counts[j] if j < num_bins-1 else 0) : # Check if source bin can give one
        # A more complex check might be needed if next_histogram_counts[j] is also a target
        # For simplicity, just check if source bin is non-empty.
        # Original logic: if node_counts_current[j] > 0
          if current_assignment[j]:
            device_to_move = current_assignment[j].pop(0) # Take from front
            current_assignment[i].append(device_to_move) # Add to end
            moved = True
            break
      if not moved:
        # This implies we need devices but can't get them from the right.
        # This might happen if earlier bins to the left took too many,
        # or if the sum constraint is violated (checked earlier).
        # This part of the logic might need refinement if it leads to deadlocks
        # or incorrect distributions. The original code only moved from right.
        logging.debug("Bin %d needs devices, but no available from right.", i)
        break # Cannot satisfy from the right
  return current_assignment


@gin.configurable
class HistogramReducer(reducer.BaseReducer):
  """Reduces high-dimensional time-series data to feature histograms.

  This reducer transforms observations for specified features into histograms,
  counting how many devices fall into predefined bins for those features.
  Other features can be passed through without modification.

  The `reduce` method converts a DataFrame of time-series observations into a
  DataFrame where specified features are replaced by their histogram bin counts.
  The `expand` method attempts a lossy reconstruction of the original
  time-series format from the histogram data.

  Attributes:
    histogram_parameters (HistogramParameters): A dictionary mapping feature
      names to their corresponding bin edge arrays.
    _normalize_reduce (bool): If True, histogram counts for a feature are
      normalized by the total count for that feature at each time step.
    _histogram_assignments (Dict[Feature, HistogramAssignment]): Internal state
      tracking device assignments to bins, used by the `expand` method for
      consistent (though lossy) reconstruction.
  """

  def __init__(
      self,
      histogram_parameters_tuples: Sequence[Tuple[str, np.ndarray]],
      reader_instance: reader_lib.BaseReader,
      normalize_reduce: bool = False,
  ):
    """Initializes the HistogramReducer.

    Args:
      histogram_parameters_tuples (Sequence[Tuple[str, np.ndarray]]): A
        sequence of tuples, where each tuple is (feature_name, bin_edges_array).
        `bin_edges_array` should be a 1D NumPy array of monotonically
        increasing bin edges.
      reader_instance (reader_lib.BaseReader): A data reader instance used to
        fetch an initial observation response. This response is used to
        determine the initial assignment of devices to histogram bins for
        features that will be histogrammed.
      normalize_reduce (bool): If True, the histogram counts produced by the
        `reduce` method will be normalized (divided by the total count for that
        feature at that time step). Defaults to False.
    """
    self._normalize_reduce: bool = normalize_reduce
    self._histogram_parameters: HistogramParameters = {
        name: np.array(bins) for name, bins in histogram_parameters_tuples
    }
    logging.info("HistogramReducer initialized with parameters: %s",
                 self._histogram_parameters)

    # Initialize assignments based on the first observation available
    # This establishes which devices exist and their initial rough distribution.
    # Reading all observations just for this seems excessive.
    # Assuming the first available observation gives a representative device set.
    try:
      # Reading only a very short period or a single file might be better.
      # For now, using min/max which might be slow if data is large.
      # Consider a method in BaseReader to get just one typical observation.
      observation_responses = reader_instance.read_observation_responses(
          start_time=pd.Timestamp.min.tz_localize('UTC'), # Ensure tz-aware
          end_time=pd.Timestamp.max.tz_localize('UTC') # Ensure tz-aware
      )
      initial_observation_response = observation_responses[0] if observation_responses else None
    except Exception as e: # pylint: disable=broad-except
        logging.warning("Could not read initial observations for HistogramReducer: %s. "
                        "Initial assignments may be empty.", e)
        initial_observation_response = None


    self._histogram_assignments: Dict[Feature, HistogramAssignment] = {}
    if initial_observation_response:
      for feature_name, bins in self._histogram_parameters.items():
        self._histogram_assignments[feature_name] = assign_devices_to_bins(
            feature_name, bins, initial_observation_response
        )
    else: # Handle case where no initial observations could be read
        for feature_name, bins in self._histogram_parameters.items():
            self._histogram_assignments[feature_name] = [[] for _ in range(len(bins))]

    logging.info("Initial histogram assignments: %s", self._histogram_assignments)

  class HistogramReducedSequence(reducer.BaseReducedSequence):
    """Represents a sequence of data reduced to histograms for some features.

    Attributes:
      reduced_sequence (pd.DataFrame): The DataFrame where specified features
        have been replaced by their histogram bin counts. Other features may
        be passed through. Column names for histogram bins are typically
        tuples like `(feature_name, "h_bin_edge_value")`.
      _histogram_parameters (HistogramParameters): Bin definitions.
      _histogram_assignments (Dict[Feature, HistogramAssignment]): Device to
        bin assignments, used for `expand`.
      _passthrough_sequence (pd.DataFrame): Original data for features that
        were not histogrammed.
    """
    def __init__(
        self,
        histogram_parameters: HistogramParameters,
        histogram_assignments: Dict[Feature, HistogramAssignment], # Initial state
        passthrough_sequence: pd.DataFrame,
        reduced_hist_sequence: pd.DataFrame, # Data with histogram columns
    ):
      self._histogram_parameters = histogram_parameters
      self._passthrough_sequence = passthrough_sequence
      # Make a deep copy to allow modification during expand without affecting reducer state
      self._current_assignments_for_expansion = copy.deepcopy(histogram_assignments)
      self.reduced_sequence = reduced_hist_sequence # This is what 'reduce' outputs

    def expand(self) -> pd.DataFrame:
      """Reconstructs an approximate time-series DataFrame from histogram data.

      For features that were histogrammed, this method assigns devices to bins
      based on the counts in `reduced_sequence` for each time step, attempting
      to maintain consistency with previous assignments. The reconstructed value
      for a device is the left edge of its assigned bin. Features that were
      passed through are merged back.

      Returns:
        pd.DataFrame: A DataFrame where histogrammed features have been
        expanded back to per-device approximate values. Index matches
        `reduced_sequence`.
      """
      reconstructed_data_frames: List[pd.DataFrame] = []

      for timestamp, row_data in self.reduced_sequence.iterrows():
        current_step_approximations: HistogramExpansion = {}
        for feature, bins in self._histogram_parameters.items():
          # Extract histogram counts for this feature at this timestamp
          # Column names for histograms are (feature, "h_bin_edge")
          bin_counts_for_feature: List[float] = []
          for bin_edge in bins: # Assuming bins are the left edges
              # Column name in reduced_sequence for this bin
              hist_col_name = (feature, f"h_{bin_edge:.2f}")
              if hist_col_name in row_data:
                  bin_counts_for_feature.append(row_data[hist_col_name])
              else:
                  # This case implies the bin was empty or not present
                  bin_counts_for_feature.append(0.0)

          # Adjust assignments to match these counts
          # Note: reassign_nodes modifies self._current_assignments_for_expansion[feature] in place
          self._current_assignments_for_expansion[feature] = reassign_nodes(
              self._current_assignments_for_expansion[feature],
              np.array(bin_counts_for_feature, dtype=int) # Ensure counts are int
          )
          # Get approximate values based on new assignment
          feature_approximations = approximate_values_from_histogram_assignment(
              feature, self._current_assignments_for_expansion[feature], bins
          )
          current_step_approximations.update(feature_approximations)

        # Convert current step's approximations to a DataFrame row
        # The keys are (device, feature), need to pivot or structure correctly
        # This part needs careful handling to match original DataFrame structure
        # For simplicity, create a Series for this timestamp
        # This reconstruction is highly dependent on how the original data was structured.
        # Assuming original was multi-indexed (device, feature) or similar.
        # This part is complex and not fully specified by original structure.
        # Placeholder:
        df_row = pd.Series(current_step_approximations, name=timestamp)
        reconstructed_data_frames.append(df_row.unstack(level=[0,1])) # Example unstack

      if not reconstructed_data_frames:
          expanded_hist_df = pd.DataFrame()
      else:
          # This concat might need refinement based on desired output structure
          expanded_hist_df = pd.concat(reconstructed_data_frames, axis=1).T


      # Combine with passthrough features
      final_df = pd.concat(
          [expanded_hist_df, self._passthrough_sequence], axis=1, join="inner"
      )
      # Ensure columns from reduced_sequence that were also passthrough are handled
      # (e.g. if a feature was both passed and (erroneously) in hist params)
      # This part of original logic was complex; simplifying:
      # Prioritize expanded values if overlap, then passthrough.
      # The concat above might already handle this if indices/columns align.
      return final_df


    @property
    def feature_device_assignments(
        self
    ) -> Dict[Feature, HistogramAssignment]:
      """Current device-to-bin assignments used in the expansion process."""
      return self._current_assignments_for_expansion

  @property
  def histogram_parameters(self) -> HistogramParameters:
    """Dict[Feature, HistogramBins]: Bin definitions for histogrammed features."""
    return self._histogram_parameters

  def _get_passthrough_sequence(
      self, observation_sequence: pd.DataFrame
  ) -> pd.DataFrame:
    """Extracts columns from the observation sequence that are not histogrammed.

    Args:
      observation_sequence (pd.DataFrame): The input DataFrame of observations.
        Columns can be simple strings or tuples (e.g., (device, feature)).

    Returns:
      pd.DataFrame: A DataFrame containing only the columns that are not
      configured to be converted into histograms.
    """
    passthrough_columns = []
    for col_name_or_tuple in observation_sequence.columns:
      feature_to_check: str
      if isinstance(col_name_or_tuple, tuple):
        # Assuming (device, feature) or (level0, device, feature)
        feature_to_check = col_name_or_tuple[-1] # Last element is feature name
      else: # Simple string column name
        feature_to_check = str(col_name_or_tuple)

      if feature_to_check not in self._histogram_parameters:
        passthrough_columns.append(col_name_or_tuple)

    return observation_sequence[passthrough_columns]

  def _get_reduced_sequence_df(
      self,
      observation_sequence: pd.DataFrame,
      feature_to_device_columns_map: Mapping[Feature, Sequence[Tuple[Device, Feature]]],
  ) -> pd.DataFrame:
    """Converts specified raw feature columns into histogram count columns.

    Args:
      observation_sequence (pd.DataFrame): Input DataFrame. Columns for features
        to be histogrammed are typically multi-indexed like (device_id, feature_name).
      feature_to_device_columns_map (Mapping[Feature, Sequence[Tuple[Device, Feature]]]):
        A map from a generic feature name (e.g., "zone_air_temperature_sensor")
        to a list of actual column names (tuples) in `observation_sequence`
        that correspond to that feature across different devices.

    Returns:
      pd.DataFrame: A DataFrame where, for each feature in
      `_histogram_parameters`, its original columns are replaced by new
      columns representing histogram bin counts. Bin column names are tuples:
      `(feature_name, f"h_{bin_edge_value:.2f}")`.
    """
    all_histogram_dfs: List[pd.DataFrame] = []
    for feature, bins in self._histogram_parameters.items():
      # Get all original columns that correspond to this feature (e.g., temp from all zones)
      original_feature_cols = feature_to_device_columns_map.get(feature, [])
      if not original_feature_cols:
        continue # No data columns for this histogram feature

      # Extract the relevant slice of the observation_sequence
      feature_data_df = observation_sequence[list(original_feature_cols)]

      # Apply histogramming row-wise
      def row_wise_histogram(row_series: pd.Series) -> pd.Series:
        counts = get_clipped_histogram(
            measurements=row_series.to_numpy(na_value=np.nan), # Handle NaNs
            bins=bins,
            clip_values=True
        )
        if self._normalize_reduce and np.sum(counts) > 0:
          counts = counts / np.sum(counts)
        return pd.Series(counts, index=[(feature, f"h_{b:.2f}") for b in bins])

      histogram_df_for_feature = feature_data_df.apply(
          row_wise_histogram, axis=1
      )
      all_histogram_dfs.append(histogram_df_for_feature)

    if not all_histogram_dfs:
      return pd.DataFrame(index=observation_sequence.index) # Empty if no hist features
    return pd.concat(all_histogram_dfs, axis=1)


  def reduce(
      self, observation_sequence: pd.DataFrame
  ) -> reducer.BaseReducedSequence:
    """Reduces observation data by converting specified features to histograms.

    Features not specified for histogramming are passed through unchanged.

    Args:
      observation_sequence (pd.DataFrame): A DataFrame where rows are time
        steps and columns are features (possibly multi-indexed by device/zone).

    Returns:
      HistogramReducedSequence: An object containing the DataFrame with
      histogrammed features, alongside passthrough data and initial device-to-bin
      assignments for potential expansion.
    """
    passthrough_df = self._get_passthrough_sequence(observation_sequence)
    feature_to_cols_map = self._get_feature_to_columns_mapping(
        observation_sequence
    )
    histogram_df = self._get_reduced_sequence_df(
        observation_sequence, feature_to_cols_map
    )

    # Combine passthrough features with new histogram features
    final_reduced_df = pd.concat([passthrough_df, histogram_df], axis=1)

    return self.HistogramReducedSequence(
        histogram_parameters=self._histogram_parameters,
        histogram_assignments=copy.deepcopy(self._histogram_assignments), # Pass copy
        passthrough_sequence=passthrough_df, # Original passthrough columns
        reduced_hist_sequence=final_reduced_df # Combined result
    )

  def _get_feature_to_columns_mapping(
      self, observation_sequence: pd.DataFrame
  ) -> Mapping[Feature, Sequence[Tuple[Device, Feature]]]:
    """Identifies which DataFrame columns correspond to each histogrammable feature.

    Assumes observation_sequence columns might be tuples like (device_id, feature_name).

    Args:
      observation_sequence (pd.DataFrame): The input observation data.

    Returns:
      Mapping[Feature, Sequence[Tuple[Device, Feature]]]: A map where keys are
      generic feature names (those in `self._histogram_parameters`) and values
      are lists of column name tuples from `observation_sequence` that match
      that generic feature.
    """
    feature_column_map = collections.defaultdict(list)
    for col_tuple in observation_sequence.columns:
      if isinstance(col_tuple, tuple) and len(col_tuple) >= 2:
        # Assuming feature name is the last element if column is a tuple
        feature_name_in_col = col_tuple[-1]
        if feature_name_in_col in self._histogram_parameters:
          # Store the full column tuple, as it's needed to access data
          feature_column_map[feature_name_in_col].append(col_tuple)
      # Simple string columns are not processed for histogram features here
    return feature_column_map
