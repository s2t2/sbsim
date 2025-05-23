"""Provides a normalizer for mapping agent actions to native setpoint values.

This module defines `BoundedActionNormalizer`, which implements the
`BaseActionNormalizer` interface. It scales actions from a normalized range
(typically [-1, 1] as used by many RL agents) to a specified native range
(e.g., temperature setpoints in Celsius or Fahrenheit).
"""

from typing import Optional # Added for type hinting

import numpy as np
from tf_agents import specs

from smart_control.models import base_normalizer

# A small tolerance for floating-point comparisons to account for precision errors.
# This allows agent actions to slightly exceed the defined normalized bounds
# without raising an error during the denormalization process.
ACTION_TOLERANCE: float = 1e-5


class BoundedActionNormalizer(base_normalizer.BaseActionNormalizer):
  """Scales normalized agent actions to a native bounded setpoint range.

  This normalizer performs a linear transformation to map action values from a
  normalized range (e.g., [-1, 1]) to a specified native range
  [`min_native_value`, `max_native_value`]. It also provides the inverse
  transformation.

  It is typically used for continuous action spaces where the agent outputs
  actions within a fixed normalized range, and these actions need to be
  converted to physical setpoint values for the environment.

  Example:
    ```python
    # Normalize agent actions from [-1, 1] to a temperature range [18C, 25C]
    temp_normalizer = BoundedActionNormalizer(
        min_native_value=18.0,
        max_native_value=25.0
    )

    # Get the action spec for the agent
    action_spec = temp_normalizer.get_array_spec(name="temperature_setpoint")
    # action_spec will be BoundedArraySpec((), np.float32, minimum=-1.0, maximum=1.0)

    # Convert agent's output (e.g., 0.5) to native setpoint value
    agent_output = np.array(0.5, dtype=np.float32)
    native_setpoint = temp_normalizer.setpoint_value(agent_output)
    # native_setpoint would be 23.25 (0.75 * (25-18) + 18)

    # Convert a native setpoint (e.g., 20C) back to agent's action space
    agent_action_val = temp_normalizer.agent_value(20.0)
    # agent_action_val would be approximately -0.428
    ```
  """

  def __init__(
      self,
      min_native_value: float,
      max_native_value: float,
      min_normalized_value: float = -1.0,
      max_normalized_value: float = 1.0,
  ):
    """Initializes the BoundedActionNormalizer.

    Args:
      min_native_value (float): The minimum value in the native (physical)
        setpoint range.
      max_native_value (float): The maximum value in the native setpoint range.
      min_normalized_value (float): The minimum value of the normalized action
        range expected from the agent. Defaults to -1.0.
      max_normalized_value (float): The maximum value of the normalized action
        range expected from the agent. Defaults to 1.0.

    Raises:
      ValueError: If `max_native_value` <= `min_native_value` or
        `max_normalized_value` <= `min_normalized_value`.
    """
    if max_native_value <= min_native_value:
      raise ValueError("max_native_value must be greater than min_native_value.")
    if max_normalized_value <= min_normalized_value:
      raise ValueError(
          "max_normalized_value must be greater than min_normalized_value."
      )

    self._min_native_value = min_native_value
    self._max_native_value = max_native_value
    self._min_normalized_value = min_normalized_value
    self._max_normalized_value = max_normalized_value
    self._native_range = max_native_value - min_native_value
    self._normalized_range = max_normalized_value - min_normalized_value

  def get_array_spec(self, name: Optional[str] = None) -> specs.BoundedArraySpec:
    """Returns the TF-Agents `BoundedArraySpec` for the normalized action.

    This spec defines the shape (scalar), dtype (float32), and bounds
    (min/max normalized value) of the action expected from the agent.

    Args:
      name (Optional[str]): An optional name for the action spec.

    Returns:
      specs.BoundedArraySpec: The action specification.
    """
    return specs.BoundedArraySpec(
        shape=(), # Scalar action
        dtype=np.float32,
        minimum=self._min_normalized_value,
        maximum=self._max_normalized_value,
        name=name or "bounded_normalized_action",
    )

  def setpoint_value(self, agent_action_value: np.ndarray) -> float:
    """Converts a normalized agent action to a native setpoint value.

    Args:
      agent_action_value (np.ndarray): A scalar NumPy array containing the
        normalized action value from the agent. Must be within the defined
        normalized bounds (plus/minus `ACTION_TOLERANCE`).

    Returns:
      float: The corresponding setpoint value in the native physical range.

    Raises:
      ValueError: If `agent_action_value` is not a scalar or is outside the
        allowed normalized bounds (considering tolerance).
    """
    if np.ndim(agent_action_value) != 0: # Must be scalar
      raise ValueError(
          f"agent_action_value must be a scalar, but received shape "
          f"{agent_action_value.shape} for value: {agent_action_value}"
      )
    # Ensure the agent action is within the expected normalized range + tolerance
    if not (
        (self._min_normalized_value - ACTION_TOLERANCE) <=
        agent_action_value <=
        (self._max_normalized_value + ACTION_TOLERANCE)
    ):
      raise ValueError(
          f"agent_action_value {agent_action_value} is outside the "
          f"normalized bounds [{self._min_normalized_value}, "
          f"{self._max_normalized_value}] (tolerance applied)."
      )
    # Clip to ensure strict adherence to bounds before transformation
    clipped_agent_action = np.clip(
        agent_action_value, self._min_normalized_value, self._max_normalized_value
    )

    # Linear scaling:
    # 1. Scale to [0, 1]: (action - min_norm) / (max_norm - min_norm)
    # 2. Scale to [min_native, max_native]: result_0_1 * native_range + min_native
    normalized_ratio = (
        (clipped_agent_action - self._min_normalized_value) /
        self._normalized_range
    )
    native_value = (
        normalized_ratio * self._native_range + self._min_native_value
    )
    return float(native_value)

  def agent_value(self, native_setpoint_value: float) -> float:
    """Converts a native setpoint value back to a normalized agent action value.

    Args:
      native_setpoint_value (float): The setpoint value in its native
        physical units. Must be within the defined native bounds.

    Returns:
      float: The corresponding normalized action value (typically in [-1, 1]).

    Raises:
      ValueError: If `native_setpoint_value` is outside the defined native
        bounds (considering `ACTION_TOLERANCE` for robustness, though ideally
        inputs should be strictly within bounds).
    """
    if not (
        (self._min_native_value - ACTION_TOLERANCE) <=
        native_setpoint_value <=
        (self._max_native_value + ACTION_TOLERANCE)
    ):
      raise ValueError(
          f"native_setpoint_value {native_setpoint_value} is outside the "
          f"native bounds [{self._min_native_value}, {self._max_native_value}] "
          f"(tolerance applied)."
      )
    # Clip to ensure strict adherence to bounds before transformation
    clipped_native_value = np.clip(
        native_setpoint_value, self._min_native_value, self._max_native_value
    )

    # Inverse linear scaling:
    # 1. Scale to [0, 1]: (native - min_native) / (max_native - min_native)
    # 2. Scale to [min_norm, max_norm]: result_0_1 * norm_range + min_norm
    native_ratio = (clipped_native_value - self._min_native_value) / self._native_range
    normalized_value = (
        native_ratio * self._normalized_range + self._min_normalized_value
    )
    return float(normalized_value)

  @property
  def setpoint_min(self) -> float:
    """float: The minimum value of the native setpoint range."""
    return self._min_native_value

  @property
  def setpoint_max(self) -> float:
    """float: The maximum value of the native setpoint range."""
    return self._max_native_value
