"""Abstract base classes for observation and action normalization.

This module defines `BaseObservationNormalizer` and `BaseActionNormalizer`,
which are abstract interfaces for transforming data between the native building
representation and the normalized representation used by the reinforcement
learning agent.

Normalization is crucial for stabilizing and speeding up RL agent training by
ensuring that input features (observations) and output signals (actions) are
within a consistent and appropriate range.

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licenses.
"""

import abc
from typing import Optional

import numpy as np
from tf_agents import specs

from smart_control.proto import smart_control_building_pb2


class BaseObservationNormalizer(metaclass=abc.ABCMeta):
  """Abstract interface for normalizing and denormalizing observations.

  Implementing classes should define how raw observation data from the
  building (e.g., sensor readings in their native units) is transformed into a
  normalized format suitable for agent input (e.g., values scaled to a
  specific range like [0, 1] or [-1, 1], or standardized to have zero mean
  and unit variance). They must also provide the inverse transformation.
  """

  @abc.abstractmethod
  def normalize(
      self, native: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Converts a raw ObservationResponse to its normalized form.

    Args:
      native (smart_control_building_pb2.ObservationResponse): The observation
        data from the building in its original, unnormalized units.

    Returns:
      smart_control_building_pb2.ObservationResponse: The observation data
      after applying the normalization transformation.
    """

  @abc.abstractmethod
  def denormalize(
      self, normalized: smart_control_building_pb2.ObservationResponse
  ) -> smart_control_building_pb2.ObservationResponse:
    """Converts a normalized ObservationResponse back to its native form.

    This is the inverse operation of `normalize`.

    Args:
      normalized (smart_control_building_pb2.ObservationResponse): The
        observation data in its normalized form.

    Returns:
      smart_control_building_pb2.ObservationResponse: The observation data
      transformed back to its original, native units.
    """


class BaseActionNormalizer(metaclass=abc.ABCMeta):
  """Abstract interface for normalizing and denormalizing agent actions.

  This class defines how actions produced by an RL agent (typically
  continuous values in a range like [-1, 1] or discrete action indices) are
  translated into meaningful setpoint values for the building (e.g., a
  temperature in Celsius or Fahrenheit). It also provides the reverse
  translation and defines the action specification for the agent.

  Conceptual Example:
    A normalizer for a continuous temperature setpoint:

    ```python
    class TemperatureSetpointNormalizer(BaseActionNormalizer):
        def __init__(self, native_min_temp_c: float, native_max_temp_c: float):
            self._min_c = native_min_temp_c
            self._max_c = native_max_temp_c
            self._range_c = native_max_temp_c - native_min_temp_c

        def get_array_spec(self, name: Optional[str] = "temp_setpoint"):
            # Agent outputs a single float between -1 and 1
            return specs.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name=name
            )

        def setpoint_value(self, agent_action: np.ndarray) -> float:
            # Convert agent action from [-1, 1] to [min_c, max_c]
            # agent_action is a 0-d array (scalar)
            normalized_value_0_1 = (agent_action.item() + 1.0) / 2.0
            return self._min_c + normalized_value_0_1 * self._range_c

        def agent_value(self, setpoint_value: float) -> float:
            # Convert native setpoint to agent action in [-1, 1]
            normalized_value_0_1 = (setpoint_value - self._min_c) / self._range_c
            return 2.0 * normalized_value_0_1 - 1.0

        @property
        def setpoint_min(self) -> float:
            return self._min_c

        @property
        def setpoint_max(self) -> float:
            return self._max_c
    ```
  """

  @abc.abstractmethod
  def get_array_spec(self, name: Optional[str] = None) -> specs.ArraySpec:
    """Defines the structure and bounds of the action for the RL agent.

    This specification informs the agent about the expected format of its
    output (e.g., shape, data type, min/max values). For instance, a
    continuous action might be a single float in [-1, 1], while a discrete
    action would have an integer type and a specific number of possible values.

    Args:
      name (Optional[str]): An optional name for the action, used to identify
        it within the agent's action specification.

    Returns:
      specs.ArraySpec: The TF-Agents specification for this action.
    """

  @abc.abstractmethod
  def setpoint_value(self, agent_action: np.ndarray) -> float:
    """Converts a normalized agent action to a native building setpoint value.

    This method takes the raw output from the agent (conforming to the
    `get_array_spec`) and transforms it into a single floating-point value
    that can be understood by the building simulation or control system (e.g.,
    a temperature in degrees Celsius).

    Args:
      agent_action (np.ndarray): The action value(s) output by the agent.
        The shape and type should match `get_array_spec()`.

    Returns:
      float: The calculated setpoint value in its native physical units.
    """

  @abc.abstractmethod
  def agent_value(self, setpoint_value: float) -> float:
    """Converts a native setpoint value to a normalized agent action value.

    This is the inverse of `setpoint_value` for cases where a native
    setpoint needs to be represented in the agent's normalized action space
    (e.g., for initializing a policy or converting default actions).
    The return type is float, implying the primary use case is for agents
    outputting continuous actions, but this might need adjustment if the
    normalizer handles discrete actions represented by floats.

    Args:
      setpoint_value (float): The setpoint value in its native physical units.

    Returns:
      float: The equivalent normalized action value for the agent, typically
      within the range defined by `get_array_spec()` (e.g., [-1, 1]).
    """

  @property
  @abc.abstractmethod
  def setpoint_min(self) -> float:
    """The minimum possible value for the native setpoint.

    Returns:
      float: The minimum valid setpoint value in native units.
    """

  @property
  @abc.abstractmethod
  def setpoint_max(self) -> float:
    """The maximum possible value for the native setpoint.

    Returns:
      float: The maximum valid setpoint value in native units.
    """
