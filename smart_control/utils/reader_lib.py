"""Abstract base classes for reading Smart Control data.

This module defines the `BaseReader` abstract class, which outlines the
interface for reading various types of data used in the Smart Control project,
such as observations, actions, rewards, and static building information
(device and zone infos). It also defines `Readers`, a utility class for managing
a collection of `BaseReader` instances.
"""

import abc
from typing import Final, Mapping, NewType, Sequence, TypeVar

from absl import logging
import gin
import pandas as pd

from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_normalization_pb2
from smart_control.proto import smart_control_reward_pb2

VariableId = NewType("VariableId", str)
"""Type alias for a string representing a unique variable identifier."""

ProtoMessageType = TypeVar("ProtoMessageType")
"""Generic type variable for protobuf message types."""


class BaseReader(metaclass=abc.ABCMeta):
  """Abstract interface for reading Smart Control project data.

  Implementations of this class are responsible for fetching various types of
  protobuf messages, typically from files or other data sources. This includes
  time-series data like observations and actions, as well as static
  configuration data like device and zone information.
  """

  @abc.abstractmethod
  def read_observation_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ObservationResponse]:
    """Reads `ObservationResponse` messages within a specified time range.

    Args:
      start_time (pd.Timestamp): The inclusive start timestamp for the data query.
      end_time (pd.Timestamp): The inclusive end timestamp for the data query.

    Returns:
      Sequence[smart_control_building_pb2.ObservationResponse]: A sequence of
      `ObservationResponse` protobuf messages.
    """

  @abc.abstractmethod
  def read_action_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_building_pb2.ActionResponse]:
    """Reads `ActionResponse` messages within a specified time range.

    Args:
      start_time (pd.Timestamp): The inclusive start timestamp for the data query.
      end_time (pd.Timestamp): The inclusive end timestamp for the data query.

    Returns:
      Sequence[smart_control_building_pb2.ActionResponse]: A sequence of
      `ActionResponse` protobuf messages.
    """

  @abc.abstractmethod
  def read_reward_infos(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardInfo]:
    """Reads `RewardInfo` messages within a specified time range.

    Args:
      start_time (pd.Timestamp): The inclusive start timestamp for the data query.
      end_time (pd.Timestamp): The inclusive end timestamp for the data query.

    Returns:
      Sequence[smart_control_reward_pb2.RewardInfo]: A sequence of `RewardInfo`
      protobuf messages.
    """

  @abc.abstractmethod
  def read_reward_responses(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp
  ) -> Sequence[smart_control_reward_pb2.RewardResponse]: # Corrected return type
    """Reads `RewardResponse` messages within a specified time range.

    Args:
      start_time (pd.Timestamp): The inclusive start timestamp for the data query.
      end_time (pd.Timestamp): The inclusive end timestamp for the data query.

    Returns:
      Sequence[smart_control_reward_pb2.RewardResponse]: A sequence of
      `RewardResponse` protobuf messages.
    """

  @abc.abstractmethod
  def read_normalization_info(
      self,
  ) -> Mapping[
      VariableId, smart_control_normalization_pb2.ContinuousVariableInfo
  ]:
    """Reads normalization parameters for continuous variables.

    Returns:
      Mapping[VariableId, ContinuousVariableInfo]: A dictionary mapping
      variable IDs to their `ContinuousVariableInfo` protobuf messages,
      which contain statistics like mean and variance.
    """

  @abc.abstractmethod
  def read_zone_infos(self) -> Sequence[smart_control_building_pb2.ZoneInfo]:
    """Reads static information about all zones in the building.

    Returns:
      Sequence[smart_control_building_pb2.ZoneInfo]: A sequence of `ZoneInfo`
      protobuf messages.
    """

  @abc.abstractmethod
  def read_device_infos(
      self,
  ) -> Sequence[smart_control_building_pb2.DeviceInfo]:
    """Reads static information about all devices in the building.

    Returns:
      Sequence[smart_control_building_pb2.DeviceInfo]: A sequence of
      `DeviceInfo` protobuf messages.
    """


@gin.configurable
class Readers:
  """A container class for managing a sequence of `BaseReader` instances.

  This class can be used to group multiple data readers, for example, if
  different types of data or data from different sources are handled by
  separate reader implementations. It is Gin-configurable, allowing the
  sequence of readers to be defined in configuration files.

  Attributes:
    readers (Sequence[BaseReader]): A sequence of `BaseReader` instances.
  """

  def __init__(self, readers: Sequence[BaseReader]):
    """Initializes the Readers collection.

    Args:
      readers (Sequence[BaseReader]): A sequence of objects that adhere to the
        `BaseReader` interface.
    """
    self._readers: Final[Sequence[BaseReader]] = readers
    logging.info("Readers manager initialized with %d reader(s).", len(readers))

  @property
  def readers(self) -> Sequence[BaseReader]:
    """Provides access to the sequence of managed `BaseReader` instances."""
    return self._readers
