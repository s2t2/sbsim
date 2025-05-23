"""Normalizes observation data using standard score (z-score) transformation.

This module provides `StandardScoreObservationNormalizer`, an implementation of
`BaseObservationNormalizer`. It normalizes observation values by subtracting
the mean and dividing by the standard deviation (square root of variance) for
each feature. The means and variances are provided via configuration.
"""

import math
from typing import Callable, Mapping, NewType

import gin

from smart_control.models import base_normalizer
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2

# Type aliases for field names, enhancing readability.
FieldNameKeyword = NewType("FieldNameKeyword", str)
"""Represents a keyword that might be part of a field name for grouping."""
FieldName = NewType("FieldName", str)
"""Represents the exact name of an observable field."""


@gin.configurable
class StandardScoreObservationNormalizer(
    base_normalizer.BaseObservationNormalizer
):
  """Normalizes and denormalizes ObservationResponse messages using z-scores.

  This normalizer applies a standard score transformation (z-score) to each
  continuous value in an `ObservationResponse`. The transformation is:
  `normalized_value = (native_value - mean) / std_dev`
  where `std_dev` is the square root of the sample variance.

  The means and variances for normalization are provided via a mapping where
  keys can be exact field names or keywords that match parts of field names
  (e.g., "temperature" to apply same normalization to all temperature sensors).
  If an exact match for a field name is not found, it falls back to keyword
  matching. If no match is found, default normalization (mean=0, std_dev=1,
  i.e., no change if variance is 1) is applied.

  Attributes:
    _normalization_constants (Mapping[FieldNameKeyword, smart_control_normalization_pb2.ContinuousVariableInfo]):
      A mapping from field name keywords (or exact field names) to their
      corresponding `ContinuousVariableInfo` protobuf messages, which store
      sample mean and variance.
  """

  def __init__(
      self,
      normalization_constants: Mapping[
          FieldNameKeyword,
          smart_control_normalization_pb2.ContinuousVariableInfo,
      ],
  ):
    """Initializes the StandardScoreObservationNormalizer.

    Args:
      normalization_constants (Mapping[FieldNameKeyword, ContinuousVariableInfo]):
        A dictionary where keys are strings (either exact field names or
        keywords like "temperature") and values are `ContinuousVariableInfo`
        protos containing `sample_mean` and `sample_variance` for those fields.
    """
    self._normalization_constants = normalization_constants

  def _get_normalization_constants(
      self, field_name: FieldName
  ) -> smart_control_normalization_pb2.ContinuousVariableInfo:
    """Retrieves normalization constants for a given field name.

    It first attempts an exact match for `field_name`. If not found, it
    iterates through the `_normalization_constants` keys to find a keyword
    that is a substring of `field_name`. If no match is found, it returns
    default constants (mean=0, variance=1) resulting in no normalization.

    Args:
      field_name (FieldName): The specific name of the field for which to get
        normalization constants.

    Returns:
      smart_control_normalization_pb2.ContinuousVariableInfo: The normalization
      constants for the field. Defaults to mean 0, variance 1 if not found.
    """
    if field_name in self._normalization_constants:
      return self._normalization_constants[field_name]

    # Fallback to keyword matching if exact field_name not found
    for keyword, constants_info in self._normalization_constants.items():
      if keyword in field_name:
        return constants_info

    # Default if no exact or keyword match is found
    return smart_control_normalization_pb2.ContinuousVariableInfo(
        id=str(field_name), sample_mean=0.0, sample_variance=1.0
    )

  def _normalize_single_value(
      self, field_name: FieldName, native_value: float
  ) -> float:
    """Normalizes a single native value using its field's mean and variance.

    Args:
      field_name (FieldName): The name of the field.
      native_value (float): The original value of the field.

    Returns:
      float: The normalized (z-score) value. Returns 0.0 if sample variance
      is zero to prevent division by zero errors.
    """
    constants_info = self._get_normalization_constants(field_name)
    if constants_info.sample_variance > 0.0:
      std_dev = math.sqrt(constants_info.sample_variance)
      return (native_value - constants_info.sample_mean) / std_dev
    # Avoid division by zero if variance is zero or negative (though should be positive)
    # In such a case, z-score is ill-defined; returning 0.0 or native_value might be options.
    # Returning 0.0 implies the normalized value is at the mean.
    return 0.0

  def _denormalize_single_value(
      self, field_name: FieldName, normalized_value: float
  ) -> float:
    """Denormalizes a single z-score value back to its native scale.

    Args:
      field_name (FieldName): The name of the field.
      normalized_value (float): The normalized (z-score) value.

    Returns:
      float: The denormalized (native) value.
    """
    constants_info = self._get_normalization_constants(field_name)
    std_dev = math.sqrt(constants_info.sample_variance) if constants_info.sample_variance > 0.0 else 1.0
    return (normalized_value * std_dev) + constants_info.sample_mean

  def normalize(
      self, native_response: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Normalizes all continuous values in an ObservationResponse.

    Args:
      native_response (smart_control_building_pb2.ObservationResponse): The
        input `ObservationResponse` with values in their native scales.

    Returns:
      smart_control_building_pb2.ObservationResponse: A new
      `ObservationResponse` message with continuous values normalized.
    """
    return self._transform_observation_response(
        native_response, self._normalize_single_value
    )

  def denormalize(
      self, normalized_response: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Denormalizes all continuous values in an ObservationResponse.

    Args:
      normalized_response (smart_control_building_pb2.ObservationResponse): The
        input `ObservationResponse` with normalized (z-score) values.

    Returns:
      smart_control_building_pb2.ObservationResponse: A new
      `ObservationResponse` message with continuous values converted back to
      their native scales.
    """
    return self._transform_observation_response(
        normalized_response, self._denormalize_single_value
    )

  def _transform_observation_response(
      self,
      input_response: smart_control_building_pb2.ObservationResponse,
      transform_function: Callable[[FieldName, float], float],
  ) -> smart_control_building_pb2.ObservationResponse:
    """Applies a given transformation function to all continuous values.

    Iterates through each `SingleObservationResponse` within the input
    `ObservationResponse` and applies `transform_function` (either
    normalization or denormalization) to its `continuous_value`.

    Args:
      input_response (smart_control_building_pb2.ObservationResponse): The
        `ObservationResponse` to transform.
      transform_function (Callable[[FieldName, float], float]): The function
        (either `_normalize_single_value` or `_denormalize_single_value`) to
        apply to each continuous value.

    Returns:
      smart_control_building_pb2.ObservationResponse: A new
      `ObservationResponse` with transformed values.
    """
    output_response = smart_control_building_pb2.ObservationResponse()
    output_response.CopyFrom(input_response) # Preserve non-transformed parts

    for single_obs_resp in output_response.single_observation_responses:
      field_name_str = (
          single_obs_resp.single_observation_request.measurement_name
      )
      original_value = single_obs_resp.continuous_value
      # Apply the transformation (normalize or denormalize)
      single_obs_resp.continuous_value = transform_function(
          FieldName(field_name_str), original_value
      )
    return output_response
