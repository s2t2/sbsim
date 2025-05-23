"""Defines a base class for simulated smart devices in an HVAC system.

This module provides the `SmartDevice` abstract base class, which serves as a
foundation for modeling various components of an HVAC system (e.g., VAVs,
boilers, air handlers). It standardizes how devices expose their observable
properties (sensors) and controllable actions (actuators) to the simulation
environment and, by extension, to a reinforcement learning agent.

The `AttributeInfo` named tuple is used to declare the mapping between
publicly visible field names (used in observation/action specs) and the
internal attribute names of the device class, along with their expected types.
"""

import abc
from typing import Any, Mapping, NamedTuple, Optional, Sequence, Type

import pandas as pd

from smart_control.proto import smart_control_building_pb2


class AttributeInfo(NamedTuple):
  """Stores metadata for an observable or actionable attribute of a SmartDevice.

  This structure links a public field name (used in observation/action specs)
  to the corresponding internal attribute name within a `SmartDevice` subclass.
  It also specifies the expected Python type of the attribute.

  Attributes:
    attribute_name (str): The actual name of the attribute (property or field)
      within the `SmartDevice` subclass.
    clazz (Type[object]): The expected Python type of the attribute (e.g.,
      `float`, `int`, `bool`).
  """
  attribute_name: str
  clazz: Type[object]


class SmartDevice(metaclass=abc.ABCMeta):
  """Abstract base class for simulated smart devices in an HVAC system.

  Subclasses of `SmartDevice` represent specific HVAC components. They must
  define their observable states (e.g., current temperature, flow rate) and
  actionable parameters (e.g., setpoints, valve positions) by providing
  mappings in the constructor.

  This class provides a standardized way to:
  - Identify the device (type, ID, zone).
  - List its observable and actionable fields.
  - Get the current value of an observable field.
  - Set the value of an actionable field.

  Attributes:
    _observable_fields (Mapping[str, AttributeInfo]): Maps public observable
      field names to their `AttributeInfo`.
    _action_fields (Mapping[str, AttributeInfo]): Maps public action field
      names to their `AttributeInfo`.
    _device_type (smart_control_building_pb2.DeviceInfo.DeviceType): The type
      of the device (e.g., AHU, VAV, BLR).
    _device_id (str): A unique identifier for this device instance.
    _zone_id (Optional[str]): Identifier of the zone this device primarily
      serves or is located in, if applicable.
    _action_timestamp (Optional[pd.Timestamp]): Timestamp of the last action
      received by this device.
    _observation_timestamp (Optional[pd.Timestamp]): Timestamp for which the
      last observation was retrieved.
  """

  def __init__(
      self,
      observable_fields: Mapping[str, AttributeInfo],
      action_fields: Mapping[str, AttributeInfo],
      device_type: smart_control_building_pb2.DeviceInfo.DeviceType,
      device_id: str,
      zone_id: Optional[str] = None,
  ):
    """Initializes a SmartDevice.

    Args:
      observable_fields (Mapping[str, AttributeInfo]): A dictionary mapping
        publicly visible names of observable fields to `AttributeInfo`
        instances that describe the corresponding internal attributes.
      action_fields (Mapping[str, AttributeInfo]): A dictionary mapping
        publicly visible names of actionable fields to `AttributeInfo`
        instances.
      device_type (smart_control_building_pb2.DeviceInfo.DeviceType): The
        category of this device (e.g., AHU, VAV).
      device_id (str): A unique string identifier for this device instance.
      zone_id (Optional[str]): The identifier of the zone this device is
        associated with. Defaults to None if not applicable.
    """
    self._observable_fields: Mapping[str, AttributeInfo] = observable_fields
    self._action_fields: Mapping[str, AttributeInfo] = action_fields
    self._device_type: smart_control_building_pb2.DeviceInfo.DeviceType = (
        device_type
    )
    self._device_id: str = device_id
    self._zone_id: Optional[str] = zone_id
    self._action_timestamp: Optional[pd.Timestamp] = None
    self._observation_timestamp: Optional[pd.Timestamp] = None

  def device_id(self) -> str:
    """Returns the unique identifier of this device."""
    return self._device_id

  def zone_id(self) -> Optional[str]:
    """Returns the identifier of the zone this device is associated with."""
    return self._zone_id

  def device_type(self) -> smart_control_building_pb2.DeviceInfo.DeviceType:
    """Returns the type of this device (e.g., AHU, VAV)."""
    return self._device_type

  def observable_field_names(self) -> Sequence[str]:
    """Returns a sequence of public names for all observable fields."""
    return list(self._observable_fields.keys())

  def action_field_names(self) -> Sequence[str]:
    """Returns a sequence of public names for all actionable fields."""
    return list(self._action_fields.keys())

  def get_observable_type(self, field_name: str) -> Type[object]:
    """Returns the expected Python type of a given observable field.

    Args:
      field_name (str): The public name of the observable field.

    Returns:
      Type[object]: The Python type (e.g., `float`, `int`) of the field.
    """
    return self._attribute_info(field_name, is_observable=True).clazz

  def get_action_type(self, field_name: str) -> Type[object]:
    """Returns the expected Python type of a given actionable field.

    Args:
      field_name (str): The public name of the actionable field.

    Returns:
      Type[object]: The Python type (e.g., `float`, `int`) of the field.
    """
    return self._attribute_info(field_name, is_observable=False).clazz

  def _attribute_info(
      self, field_name: str, *, is_observable: bool
  ) -> AttributeInfo:
    """Retrieves `AttributeInfo` for a field and validates its existence.

    Args:
      field_name (str): The public name of the field.
      is_observable (bool): True if querying an observable field, False if
        querying an actionable field.

    Returns:
      AttributeInfo: The attribute information for the specified field.

    Raises:
      AttributeError: If `field_name` is not declared as the specified type
        (observable/actionable) or if its mapped internal attribute does not
        exist on the device instance.
    """
    field_map = self._observable_fields if is_observable else self._action_fields
    field_type_str = "observable" if is_observable else "action"

    if field_name not in field_map:
      raise AttributeError(
          f"Field '{field_name}' is not declared as an {field_type_str} "
          f"field for device '{self._device_id}'. Available "
          f"{field_type_str} fields: {list(field_map.keys())}"
      )

    attr_info = field_map[field_name]
    if not hasattr(self, attr_info.attribute_name):
      raise AttributeError(
          f"Internal attribute '{attr_info.attribute_name}' for field "
          f"'{field_name}' does not exist on device '{self._device_id}'."
      )
    return attr_info

  def get_observation(
      self, observable_field_name: str, observation_timestamp: pd.Timestamp
  ) -> Any:
    """Retrieves the current value of a specified observable field.

    Args:
      observable_field_name (str): The public name of the observable field
        whose value is to be retrieved.
      observation_timestamp (pd.Timestamp): The timestamp for which this
        observation is being requested. This is stored internally.

    Returns:
      Any: The current value of the specified observable field.

    Raises:
      AttributeError: If `observable_field_name` is not a declared observable
        field or its mapped attribute doesn't exist.
    """
    attr_info = self._attribute_info(observable_field_name, is_observable=True)
    self._observation_timestamp = observation_timestamp
    # getattr is used to access the property or attribute dynamically
    return getattr(self, attr_info.attribute_name)

  def set_action(
      self, action_field_name: str, value: Any, action_timestamp: pd.Timestamp
  ) -> None:
    """Sets the value of a specified actionable field on the device.

    Args:
      action_field_name (str): The public name of the actionable field.
      value (Any): The new value to set for the field.
      action_timestamp (pd.Timestamp): The timestamp when this action is
        applied. This is stored internally.

    Raises:
      AttributeError: If `action_field_name` is not a declared actionable
        field or its mapped attribute doesn't exist.
      ValueError: If the provided `value` is not of the expected type for
        the field.
    """
    attr_info = self._attribute_info(action_field_name, is_observable=False)
    self._action_timestamp = action_timestamp

    if not isinstance(value, attr_info.clazz):
      raise ValueError(
          f"Cannot set action field '{action_field_name}' on device "
          f"'{self._device_id}': expected type {attr_info.clazz}, "
          f"but got type {type(value)} for value '{value}'."
      )
    # setattr is used to modify the property or attribute dynamically
    setattr(self, attr_info.attribute_name, value)
