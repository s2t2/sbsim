"""Implements a time-based schedule policy for controlling building setpoints.

This module defines `SchedulePolicy`, a TF-Agents policy that determines
actions (setpoint values) based on predefined schedules for weekdays and
weekends/holidays. It uses time features from the environment's observation
to select the appropriate setpoint value from the configured schedules.
"""

import dataclasses
import enum
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import policy_step
from tf_agents.typing import types

from smart_control.models import base_normalizer # For ActionNormalizerMap type hint
from smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE
from smart_control.reinforcement_learning.utils.time_utils import to_dow
from smart_control.reinforcement_learning.utils.time_utils import to_hod

logger = logging.getLogger(__name__)


class DeviceType(enum.Enum):
  """Enumerates controllable device types relevant to the schedule policy.

  Used to categorize devices in `ScheduleEvent` and `ActionSequence`.
  """
  AC = 0  # Air Conditioning unit
  HWS = 1 # Hot Water System/Supply


# Type Aliases
SetpointName = str
"""Type alias for the name of a controllable setpoint (e.g., 'supply_air_temp')."""

SetpointValue = Union[float, int, bool]
"""Type alias for the value of a setpoint."""

ActionSequence = List[Tuple[DeviceType, SetpointName]]
"""Type alias for an ordered list of (DeviceType, SetpointName) tuples,
defining the structure of actions this policy will generate."""

ActionNormalizerMap = Dict[Any, base_normalizer.BaseActionNormalizer]
"""Type alias for a map from normalizer keys to action normalizer instances.
The key type is `Any` as it depends on how normalizers are keyed in the env.
"""

@dataclasses.dataclass
class ScheduleEvent:
  """Represents a single event in a schedule, setting a value at a time.

  Attributes:
    start_time (pd.Timedelta): The time offset from the beginning of the day
      (e.g., `pd.Timedelta(6, unit='hour')` for 6 AM) when this setpoint
      value becomes active.
    device (DeviceType): The type of device this event applies to.
    setpoint_name (SetpointName): The specific setpoint on the device to control.
    setpoint_value (SetpointValue): The native value to apply to the setpoint.
  """
  start_time: pd.Timedelta
  device: DeviceType
  setpoint_name: SetpointName
  setpoint_value: SetpointValue


Schedule = List[ScheduleEvent]
"""Type alias for a schedule, which is a list of `ScheduleEvent` objects."""


def get_active_setpoint(
    schedule: Schedule,
    device: DeviceType,
    setpoint_name: SetpointName,
    current_time_offset: pd.Timedelta,
) -> Optional[SetpointValue]:
  """Finds the currently active setpoint value from a schedule.

  Given a schedule, device type, setpoint name, and the current time offset
  from the start of the day, this function determines the setpoint value that
  should be active. It selects the value from the latest event in the schedule
  that occurred at or before the `current_time_offset`. If no such event exists
  (e.g., current time is before the first scheduled event), it wraps around and
  uses the value from the last event of the previous day's schedule.

  Args:
    schedule (Schedule): A list of `ScheduleEvent` objects defining the
      setpoint changes throughout a day.
    device (DeviceType): The type of device to query.
    setpoint_name (SetpointName): The name of the setpoint to query.
    current_time_offset (pd.Timedelta): The current time as an offset from the
      start of the day (e.g., time since midnight).

  Returns:
    Optional[SetpointValue]: The active setpoint value. Returns None if no
    matching events are found in the schedule for the given device/setpoint,
    or if the schedule itself is empty for that device/setpoint.
  """
  # Filter events for the specific device and setpoint
  relevant_events = {
      event.start_time: event.setpoint_value
      for event in schedule
      if event.device == device and event.setpoint_name == setpoint_name
  }

  if not relevant_events:
    logger.warning(
        "No schedule events found for device %s, setpoint %s.",
        device,
        setpoint_name,
    )
    return None # No configuration for this specific device/setpoint

  # Convert to a Pandas Series for easier time-based lookup and sorting
  # The index will be the Timedelta start_time of events.
  event_series = pd.Series(relevant_events).sort_index()

  # Find events that happened at or before the current_time_offset
  # `searchsorted` finds where `current_time_offset` would be inserted to
  # maintain order. `side='right'` means it gives the index of the first
  # element strictly greater than `current_time_offset`.
  # So, index - 1 gives the last event at or before `current_time_offset`.
  idx = event_series.index.searchsorted(current_time_offset, side="right")

  if idx == 0:
    # current_time_offset is before the first event in the schedule.
    # Wrap around: use the value from the last event of the day.
    return event_series.iloc[-1]
  else:
    # Select the event at index idx - 1.
    return event_series.iloc[idx - 1]


class SchedulePolicy(tf_policy.TFPolicy):
  """A TF-Agents policy that determines actions based on predefined schedules.

  This policy uses the time of day and day of week (extracted from the
  environment's observation) to look up setpoint values in separate schedules
  for weekdays and weekends/holidays. The actions are then normalized using
  provided normalizers before being output.

  Attributes:
    weekday_schedule (Schedule): List of `ScheduleEvent`s for weekdays.
    weekend_schedule (Schedule): List of `ScheduleEvent`s for weekends/holidays.
    dow_sin_index (int): Index of the 'day of week sine' feature in the
      observation tensor.
    dow_cos_index (int): Index of the 'day of week cosine' feature.
    hod_sin_index (int): Index of the 'hour of day sine' feature.
    hod_cos_index (int): Index of the 'hour of day cosine' feature.
    action_sequence (ActionSequence): Ordered list of (DeviceType, SetpointName)
      tuples defining the structure of the output action.
    action_normalizers (ActionNormalizerMap): Dictionary mapping normalizer keys
      (often DeviceFieldId or setpoint names) to `BaseActionNormalizer` objects.
    local_start_time (pd.Timestamp): The local start time of the episode, used
      as a reference for calculating current time offsets.
    norm_mean (float): Mean used for denormalizing time features (assumed 0).
    norm_std (float): Standard deviation for denormalizing time features (assumed 1).
  """

  def __init__(
      self,
      time_step_spec: types.TimeStep,
      action_spec: types.NestedTensorSpec,
      action_sequence: ActionSequence,
      weekday_schedule: Schedule,
      weekend_schedule: Schedule,
      dow_sin_index: int,
      dow_cos_index: int,
      hod_sin_index: int,
      hod_cos_index: int,
      action_normalizers: ActionNormalizerMap,
      local_start_time: pd.Timestamp,
      name: Optional[str] = "SchedulePolicy",
  ):
    """Initializes the SchedulePolicy.

    Args:
      time_step_spec (types.TimeStep): Spec of the time_steps.
      action_spec (types.NestedTensorSpec): Spec of the actions.
      action_sequence (ActionSequence): Defines the order and types of actions
        this policy will produce.
      weekday_schedule (Schedule): Schedule to use on weekdays.
      weekend_schedule (Schedule): Schedule to use on weekends/holidays.
      dow_sin_index (int): Index of 'day of week sine' in observation.
      dow_cos_index (int): Index of 'day of week cosine' in observation.
      hod_sin_index (int): Index of 'hour of day sine' in observation.
      hod_cos_index (int): Index of 'hour of day cosine' in observation.
      action_normalizers (ActionNormalizerMap): Normalizers for converting
        native setpoint values to agent action values.
      local_start_time (pd.Timestamp): The local start timestamp of the
        current episode, used to correctly interpret time offsets.
      name (Optional[str]): A name for this policy.
    """
    self.weekday_schedule = weekday_schedule
    self.weekend_schedule = weekend_schedule
    self.dow_sin_index = dow_sin_index
    self.dow_cos_index = dow_cos_index
    self.hod_sin_index = hod_sin_index
    self.hod_cos_index = hod_cos_index
    self.action_sequence = action_sequence
    self.action_normalizers = action_normalizers
    self.local_start_time = local_start_time
    # Assuming time features in observation are normalized with mean 0, std 1.
    self.norm_mean: float = 0.0
    self.norm_std: float = 1.0

    super().__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=(), # This policy is stateless
        info_spec=(),
        clip=False, # Actions are assumed to be handled by normalizers
        observation_and_action_constraint_splitter=None,
        name=name,
    )

  def _normalize_actions(
      self, action_map: Dict[Tuple[DeviceType, SetpointName], SetpointValue]
  ) -> Dict[Tuple[DeviceType, SetpointName], float]:
    """Normalizes native setpoint values to agent action values.

    Args:
      action_map (Dict[Tuple[DeviceType, SetpointName], SetpointValue]): A
        dictionary mapping (device, setpoint_name) tuples to their native
        setpoint values determined by the schedule.

    Returns:
      Dict[Tuple[DeviceType, SetpointName], float]: A dictionary mapping
      (device, setpoint_name) to their normalized action values (typically
      in [-1, 1]).

    Raises:
      KeyError: If a normalizer cannot be found for a given setpoint name.
    """
    normalized_actions: Dict[Tuple[DeviceType, SetpointName], float] = {}
    for (device, setpoint_name), native_value in action_map.items():
      normalizer_found = False
      # The normalizer keys might be DeviceFieldId strings (e.g., "AC_supply_air_temp")
      # or just SetpointName. This loop tries to find a match.
      for normalizer_key, normalizer_instance in self.action_normalizers.items():
        # Heuristic: check if the normalizer_key (string) ends with the setpoint_name.
        # This assumes normalizer keys are consistently named.
        if isinstance(normalizer_key, str) and normalizer_key.endswith(setpoint_name):
          normalized_actions[(device, setpoint_name)] = normalizer_instance.agent_value(
              float(native_value) # Ensure value is float for normalizer
          )
          normalizer_found = True
          break
      if not normalizer_found:
        raise KeyError(
            f"No action normalizer found for setpoint: {setpoint_name} "
            f"(device: {device}). Available normalizer keys: "
            f"{list(self.action_normalizers.keys())}"
        )
    return normalized_actions

  def _get_action_map(
      self, time_step: types.TimeStep
  ) -> Dict[Tuple[DeviceType, SetpointName], Optional[SetpointValue]]:
    """Determines native setpoint values based on the current time.

    Args:
      time_step (types.TimeStep): The current time_step from the environment,
        containing the observation with time features.

    Returns:
      Dict[Tuple[DeviceType, SetpointName], Optional[SetpointValue]]: A map from
      (device, setpoint_name) to the native setpoint value determined by the
      active schedule. Values can be None if not found in schedule.
    """
    observation = time_step.observation # This is a batched observation

    # Assuming observation is shaped [batch_size, num_features]
    # We take the first element of the batch for time feature extraction.
    # If batch_size > 1, this policy will apply the same schedule-based
    # action to all elements in the batch.
    obs_flat = observation[0]

    # Denormalize time features (assuming they were normalized with mean 0, std 1)
    dow_sin = (obs_flat[self.dow_sin_index] * self.norm_std) + self.norm_mean
    dow_cos = (obs_flat[self.dow_cos_index] * self.norm_std) + self.norm_mean
    hod_sin = (obs_flat[self.hod_sin_index] * self.norm_std) + self.norm_mean
    hod_cos = (obs_flat[self.hod_cos_index] * self.norm_std) + self.norm_mean

    # Convert sine/cosine pairs back to day of week (0-6) and hour of day (0-23)
    day_of_week = to_dow(dow_sin.numpy(), dow_cos.numpy())
    hour_of_day = to_hod(hod_sin.numpy(), hod_cos.numpy())

    # Determine current time offset from midnight for schedule lookup
    # `local_start_time.utcoffset()` gives the UTC offset of the local time zone.
    # This seems to adjust hour_of_day to be relative to UTC midnight if
    # schedules are defined in a local timezone context but hod is UTC-based.
    # Simpler: assume hod is local hour, schedules use local time offsets.
    current_time_offset = pd.Timedelta(hour_of_day, unit="hour")
    # If schedules are based on local time, this is what we need.
    # The original code `+ self.local_start_time.utcoffset()` is confusing
    # if `hour_of_day` is already local. Assuming `hour_of_day` is local.

    # Select appropriate schedule (weekday or weekend/holiday)
    # Monday is 0, Sunday is 6 for `day_of_week` from `to_dow`.
    active_schedule = (
        self.weekday_schedule if day_of_week < 5 else self.weekend_schedule
    )

    # Get the active setpoint for each action defined in action_sequence
    action_values: Dict[Tuple[DeviceType, SetpointName], Optional[SetpointValue]] = {}
    for device, setpoint in self.action_sequence:
      action_values[(device, setpoint)] = get_active_setpoint(
          active_schedule, device, setpoint, current_time_offset
      )
    return action_values

  def _action(
      self,
      time_step: types.TimeStep,
      policy_state: types.NestedTensor, # Unused for this stateless policy
      seed: Optional[types.Seed] = None, # Unused, policy is deterministic
  ) -> policy_step.PolicyStep:
    """Computes the action for the given TimeStep.

    Args:
      time_step (types.TimeStep): The current TimeStep from the environment.
      policy_state (types.NestedTensor): The previous policy state (unused).
      seed (Optional[types.Seed]): Random seed for stochastic policies (unused).

    Returns:
      policy_step.PolicyStep: A PolicyStep object containing the action,
      empty state, and empty info.

    Raises:
      ValueError: If a setpoint value from the schedule is None after lookup.
    """
    del policy_state, seed # Mark as unused

    native_action_map = self._get_action_map(time_step)
    # Check for None values which indicate missing schedule entries
    for key, value in native_action_map.items():
        if value is None:
            raise ValueError(
                f"Setpoint value for {key} not found in schedule. "
                "Ensure schedules cover all (DeviceType, SetpointName) "
                "pairs in `action_sequence`."
            )

    # Type assertion for mypy after check, as values are no longer Optional.
    checked_native_action_map = typing.cast(
        Dict[Tuple[DeviceType, SetpointName], SetpointValue],
        native_action_map
    )
    normalized_action_map = self._normalize_actions(checked_native_action_map)


    # Convert the map of normalized actions to an ordered array
    action_list = []
    for device, setpoint in self.action_sequence:
        action_list.append(normalized_action_map[(device, setpoint)])
    action_array = np.array(action_list, dtype=np.float32)

    # Add batch dimension as TF-Agents policies expect batched actions
    # Action spec is typically (num_actions,), policy outputs [1, num_actions]
    batched_action_array = np.expand_dims(action_array, axis=0)
    action_tensor = tf.convert_to_tensor(batched_action_array)

    return policy_step.PolicyStep(action=action_tensor, state=(), info=())


def create_baseline_schedule_policy(
    tf_env: tf_py_environment.TFPyEnvironment,
) -> SchedulePolicy:
  """Creates a baseline `SchedulePolicy` with predefined schedules.

  This function configures a `SchedulePolicy` with a default set of schedules
  for air conditioning and hot water systems. It serves as a benchmark or
  a starting point for data collection.

  The schedules are:
  - Weekdays:
    - AC Supply Air Heating Setpoint: 292.0K (18.85C) from 6 AM to 7 PM,
      285.0K (11.85C) otherwise.
    - HWS Supply Water Setpoint: 350.0K (76.85C) from 6 AM to 7 PM,
      315.0K (41.85C) otherwise.
  - Weekends/Holidays:
    - AC Supply Air Heating Setpoint: 285.0K (11.85C) all day.
    - HWS Supply Water Setpoint: 315.0K (41.85C) all day.

  Args:
    tf_env (tf_py_environment.TFPyEnvironment): The TensorFlow environment wrapper.
      This is used to extract action specs, time step specs, observation feature
      indices, action normalizers, and the initial simulation time.

  Returns:
    SchedulePolicy: A configured instance of `SchedulePolicy`.
  """
  # Assumes tf_env wraps a single Python environment
  py_env = tf_env.pyenv.envs[0]

  # Extract necessary specs and info from the environment
  observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(tf_env)

  # Get indices for time features from the observation spec (field_names)
  # This relies on the environment having these specific field names.
  try:
    hod_cos_index = py_env.field_names.index("hod_cos_000")
    hod_sin_index = py_env.field_names.index("hod_sin_000")
    dow_cos_index = py_env.field_names.index("dow_cos_000")
    dow_sin_index = py_env.field_names.index("dow_sin_000")
  except ValueError as e:
    raise ValueError(
        "One or more required time feature names (hod_cos_000, etc.) not "
        f"found in environment's field_names: {py_env.field_names}"
    ) from e

  # Define schedules (temperatures are in Kelvin)
  weekday_schedule_events: Schedule = [
      ScheduleEvent(
          start_time=pd.Timedelta(6, unit="hour"),
          device=DeviceType.AC,
          setpoint_name="supply_air_heating_temperature_setpoint",
          setpoint_value=292.0, # Approx. 18.85°C
      ),
      ScheduleEvent(
          start_time=pd.Timedelta(19, unit="hour"),
          device=DeviceType.AC,
          setpoint_name="supply_air_heating_temperature_setpoint",
          setpoint_value=285.0, # Approx. 11.85°C
      ),
      ScheduleEvent(
          start_time=pd.Timedelta(6, unit="hour"),
          device=DeviceType.HWS,
          setpoint_name="supply_water_setpoint",
          setpoint_value=350.0, # Approx. 76.85°C
      ),
      ScheduleEvent(
          start_time=pd.Timedelta(19, unit="hour"),
          device=DeviceType.HWS,
          setpoint_name="supply_water_setpoint",
          setpoint_value=315.0, # Approx. 41.85°C
      ),
  ]

  weekend_holiday_schedule_events: Schedule = [
      ScheduleEvent(
          start_time=pd.Timedelta(0, unit="hour"), # Active all day
          device=DeviceType.AC,
          setpoint_name="supply_air_heating_temperature_setpoint",
          setpoint_value=285.0,
      ),
      # Redundant event for AC, as 00:00 covers the whole day if no other event
      # ScheduleEvent(
      #     pd.Timedelta(19, unit='hour'), DeviceType.AC,
      #     'supply_air_heating_temperature_setpoint', 285.0
      # ),
      ScheduleEvent(
          start_time=pd.Timedelta(0, unit="hour"), # Active all day
          device=DeviceType.HWS,
          setpoint_name="supply_water_setpoint",
          setpoint_value=315.0,
      ),
      # Redundant event for HWS
      # ScheduleEvent(
      #     pd.Timedelta(19, unit='hour'), DeviceType.HWS,
      #     'supply_water_setpoint', 315.0
      # ),
  ]

  # The policy needs the local start time of the episode for reference.
  # Assuming current_simulation_timestamp is UTC.
  local_start_time = py_env.current_simulation_timestamp.tz_convert(
      DEFAULT_TIME_ZONE
  )

  # Define the sequence of actions the policy will output, matching action_spec
  # This order must be consistent with how the agent's actions are structured.
  defined_action_sequence: ActionSequence = [
      (DeviceType.AC, "supply_air_heating_temperature_setpoint"),
      (DeviceType.HWS, "supply_water_setpoint"),
  ]

  policy = SchedulePolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      action_sequence=defined_action_sequence,
      weekday_schedule=weekday_schedule_events,
      weekend_schedule=weekend_holiday_schedule_events,
      action_normalizers=py_env.action_normalizers,
      hod_cos_index=hod_cos_index,
      hod_sin_index=hod_sin_index,
      dow_cos_index=dow_cos_index,
      dow_sin_index=dow_sin_index,
      local_start_time=local_start_time,
      name="BaselineSchedulePolicy",
  )

  return policy
