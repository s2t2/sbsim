"""Defines observation and action normalizer base classes.

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licenses.
"""

import abc

import numpy as np
from tf_agents import specs

from smart_control.proto import smart_control_building_pb2


class BaseObservationNormalizer(metaclass=abc.ABCMeta):
  """Abstract base class for observation normalization.

  This class defines the interface for normalizing raw observations from a
  building environment into a format suitable for a reinforcement learning
  agent, and for denormalizing them back to their original scale.

  Implementations might perform various transformations, such as:
  - Min-max scaling
  - Z-score standardization
  - Applying offsets
  """

  @abc.abstractmethod
  def normalize(
      self, native: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Normalizes a raw `ObservationResponse` from the building.

    Args:
      native: An `ObservationResponse` protobuf message containing observation
        values in their original, physical units (e.g., degrees Celsius, ppm).

    Returns:
      An `ObservationResponse` protobuf message where the observation values
      have been transformed into a normalized representation (e.g., scaled
      to a [0, 1] or [-1, 1] range, or standardized).
    """

  @abc.abstractmethod
  def denormalize(
      self, normalized: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Denormalizes an `ObservationResponse` back to native physical units.

    This is the inverse operation of `normalize`. It's useful for interpreting
    normalized agent outputs or for converting stored normalized data back
    to its original scale.

    Args:
      normalized: An `ObservationResponse` protobuf message containing
        normalized observation values.

    Returns:
      An `ObservationResponse` protobuf message where the observation values
      have been transformed back to their original physical units.
    """


class BaseActionNormalizer(metaclass=abc.ABCMeta):
  """Abstract base class for action normalization and specification.

  This class defines the interface for:
  1.  Converting actions from an RL agent (typically normalized, e.g., in the
      range [-1, 1] or as discrete indices) into physical setpoint values
      that can be applied to a building system.
  2.  Converting physical setpoint values back into the agent's action format.
  3.  Providing a `tf_agents.specs.ArraySpec` that defines the structure,
      data type, and bounds of the action expected from the agent for a
      specific setpoint.
  4.  Exposing the valid range (min and max) of the physical setpoint.
  """

  @abc.abstractmethod
  def get_array_spec(self, name: str | None = None) -> specs.ArraySpec:
    """Returns the `tf_agents.specs.ArraySpec` for this action.

    The spec defines the shape, data type, and bounds of the action that the
    RL agent should produce. For example, a continuous setpoint might have a
    spec like `BoundedArraySpec(shape=(), dtype=float32, minimum=-1.0, maximum=1.0)`,
    while a discrete setpoint (e.g., choosing from N levels) might be
    `DiscreteArraySpec(shape=(), dtype=int32, num_values=N, name=name)`.

    Args:
      name: An optional string name for the action spec. This can be useful
        for debugging or identifying specs in more complex action structures.

    Returns:
      A `tf_agents.specs.ArraySpec` (or a subclass like `BoundedArraySpec`,
      `DiscreteArraySpec`) defining the agent's action space for this setpoint.
    """

  @abc.abstractmethod
  def setpoint_value(self, agent_action: np.ndarray) -> float:
    """Converts a normalized agent action into a physical setpoint value.

    Implementations will take the raw output from the agent (which conforms to
    the `ArraySpec` from `get_array_spec`) and transform it into a single
    floating-point number representing the value to be applied to the
    building's setpoint (e.g., target temperature, damper position).

    For example, if the agent outputs a value in [-1, 1], this method would
    scale and offset it to the setpoint's native range (e.g., 18.0°C to 25.0°C).
    If the action is discrete (e.g., an integer index), this method would map
    that index to a specific physical value.

    Args:
      agent_action: A NumPy array representing the action from the RL agent.
        The structure of this array must match the `ArraySpec` returned by
        `get_array_spec()`. For continuous actions, this is often a scalar
        array (e.g., `np.array(0.5)`). For discrete actions, it might be an
        integer index (e.g., `np.array(2)`).

    Returns:
      The calculated physical setpoint value in its native engineering units.
    """

  @abc.abstractmethod
  def agent_value(self, setpoint_value: float) -> np.ndarray:
    """Converts a physical setpoint value into the agent's action format.

    This is the inverse of `setpoint_value`. It takes a value in native
    engineering units and transforms it into the normalized representation
    that the agent uses or would output. This can be useful for:
    - Setting an initial policy for the agent based on current building setpoints.
    - Converting human-defined setpoints into the agent's action space for
      analysis or as default actions.

    Note: The base class signature returns `float`, but implementations for
    actions requiring an `np.ndarray` (e.g., one-hot encoded discrete actions)
    should return `np.ndarray`. The return type here will be updated to
    `np.ndarray` for consistency with `agent_action` in `setpoint_value`.

    Args:
      setpoint_value: The physical setpoint value in its native units.

    Returns:
      A NumPy array representing the action in the agent's normalized format,
      compatible with the `ArraySpec` from `get_array_spec()`.
    """

  @property
  @abc.abstractmethod
  def setpoint_min(self) -> float:
    """The minimum valid physical value for the setpoint.

    Returns:
      The minimum setpoint value in native engineering units (e.g., degrees
      Celsius, percentage).
    """

  @property
  @abc.abstractmethod
  def setpoint_max(self) -> float:
    """The maximum valid physical value for the setpoint.

    Returns:
      The maximum setpoint value in native engineering units (e.g., degrees
      Celsius, percentage).
    """
