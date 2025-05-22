"""Defines a TF-Agents policy that operates based on predefined time schedules.

This module provides `SchedulePolicy`, a policy that determines actions by
looking up setpoint values from schedules (e.g., different schedules for
weekdays and weekends). It uses time features (day of week, hour of day)
extracted from observations to select the appropriate action from the configured
schedules.

Supporting dataclasses and enums like `DeviceType` and `ScheduleEvent` are
also defined to structure the schedule information. A helper function
`get_active_setpoint` is used to retrieve the correct setpoint value for a
given time. Additionally, a factory function `create_baseline_schedule_policy`
is provided to instantiate a `SchedulePolicy` with a default baseline schedule.
"""

import dataclasses
import enum
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union # Added Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import tf_policy
from tf_agents.specs import tensor_spec # For type hints
from tf_agents.train.utils import spec_utils
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts # For type hints
from tf_agents.typing import types

from smart_control.models import base_normalizer # For type hinting action_normalizers
from smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE
from smart_control.reinforcement_learning.utils.time_utils import to_dow
from smart_control.reinforcement_learning.utils.time_utils import to_hod

logger = logging.getLogger(__name__)


class DeviceType(enum.Enum):
  """Enumerates the types of devices controllable by the schedule.

  Attributes:
    AC: Air Conditioning unit.
    HWS: Hot Water System.
  """
  AC = 0
  HWS = 1


# Type aliases for clarity
SetpointName = str
SetpointValue = Union[float, int, bool]
# An ordered list defining which (DeviceType, SetpointName) pairs correspond
# to which elements in the output action tensor.
ActionSequence = Sequence[Tuple[DeviceType, SetpointName]]


@dataclasses.dataclass
class ScheduleEvent:
  """Represents a single event within a time-based schedule.

  A schedule event defines a specific setpoint value that should be applied to a
  particular device and setpoint name, starting from a given time of day.

  Attributes:
    start_time: A `pd.Timedelta` indicating the time of day (e.g., from
      midnight) when this setpoint value becomes active.
    device: The `DeviceType` to which this event applies.
    setpoint_name: The string identifier of the setpoint to be changed
      (e.g., 'supply_air_heating_temperature_setpoint').
    setpoint_value: The actual value (float, int, or bool) to apply to the
      setpoint.
  """
  start_time: pd.Timedelta
  device: DeviceType
  setpoint_name: SetpointName
  setpoint_value: SetpointValue


# A schedule is a list of ScheduleEvents, typically ordered by start_time.
Schedule = List[ScheduleEvent]


def get_active_setpoint(
    schedule: Schedule,
    device: DeviceType,
    setpoint_name: SetpointName,
    timestamp: pd.Timedelta,
) -> Optional[SetpointValue]:
  """Finds the active setpoint value from a schedule for a given time.

  This function iterates through a schedule (a list of `ScheduleEvent` objects)
  to find the most recent event that matches the specified `device` and
  `setpoint_name` and occurred at or before the given `timestamp` (time since
  midnight).

  If no event is found at or before the current `timestamp` within the schedule
  (e.g., if the `timestamp` is before the first event of the day), it "wraps
  around" and returns the value from the last event in the schedule. This implies
  that the setpoint from the end of the previous day carries over.

  Args:
    schedule: A list of `ScheduleEvent` objects representing the time-based
      schedule for setpoints.
    device: The `DeviceType` (e.g., `DeviceType.AC`) to filter events for.
    setpoint_name: The string name of the setpoint to find (e.g.,
      'supply_air_heating_temperature_setpoint').
    timestamp: A `pd.Timedelta` representing the current time of day (offset
      from midnight) for which to find the active setpoint.

  Returns:
    The `SetpointValue` from the active schedule event. Returns `None` if no
    matching events are found in the schedule (which typically shouldn't happen
    if schedules are well-defined).
  """
  logger.debug('Getting active setpoint for %s - %s at %s', device, setpoint_name, timestamp)

  # Create a dictionary of {time: value} for the specific device and setpoint
  # Events are assumed to be sorted by time if multiple exist for the same time,
  # though dict creation itself doesn't guarantee order if keys are identical.
  # However, pd.Series sorts by index later.
  events = {
      event.start_time: event.setpoint_value
      for event in schedule
      if event.device == device and event.setpoint_name == setpoint_name
  }

  if not events:
    logger.warning('No events found for device %s, setpoint %s in schedule.', device, setpoint_name)
    return None

  # Convert to Series for easier time-based lookup; Series sorts by index (time)
  series = pd.Series(events).sort_index()

  # Find events that happened at or before the timestamp
  # series.index is a TimedeltaIndex. series.index <= timestamp performs element-wise comparison.
  prior_event_indices = series.index[series.index <= timestamp]

  if prior_event_indices.empty:
    # If no prior events today (e.g., current time is before the first scheduled event),
    # use the last event from the schedule (implies carry-over from previous day's end).
    logger.debug("No prior events for today, using last event from schedule.")
    return series.iloc[-1]
  else:
    # Use the value from the most recent event at or before the current timestamp
    active_event_time = prior_event_indices[-1]
    logger.debug("Active event time: %s", active_event_time)
    return series.loc[active_event_time]


class SchedulePolicy(tf_policy.TFPolicy):
  """A TF-Agents policy that selects actions based on time-dependent schedules.

  This policy implements a rule-based approach where actions are determined by
  predefined schedules for weekdays and weekends/holidays. It extracts time
  information (day of week, hour of day) from the observation, identifies the
  active setpoints from the appropriate schedule, normalizes these setpoints
  into the agent's action space, and constructs the action tensor.

  It does not involve any learning from environment interactions but serves as a
  deterministic, time-driven baseline or rule-based controller.
  """

  def __init__(
      self,
      time_step_spec: ts.TimeStepSpec,
      action_spec: tensor_spec.BoundedTensorSpec, # More specific type
      action_sequence: ActionSequence,
      weekday_schedule: Schedule,
      weekend_schedule: Schedule,
      dow_sin_index: int,
      dow_cos_index: int,
      hod_sin_index: int,
      hod_cos_index: int,
      action_normalizers: Dict[str, base_normalizer.BaseActionNormalizer],
      local_start_time: pd.Timestamp,
      name: Optional[str] = None,
  ):
    """Initializes the SchedulePolicy.

    Args:
      time_step_spec: A `tf_agents.trajectories.time_step.TimeStepSpec` defining
        the observations provided by the environment.
      action_spec: A `tf_agents.specs.tensor_spec.BoundedTensorSpec` defining
        the structure and bounds of the actions to be produced by the policy.
      action_sequence: An ordered sequence of `(DeviceType, SetpointName)`
        tuples. This defines the mapping from scheduled device setpoints to
        the elements of the output action tensor. The order in this sequence
        determines the order in the final action array.
      weekday_schedule: A list of `ScheduleEvent` objects defining the setpoint
        schedule for weekdays.
      weekend_schedule: A list of `ScheduleEvent` objects defining the setpoint
        schedule for weekends and holidays.
      dow_sin_index: The index in the observation tensor corresponding to the
        sine component of the day of the week encoding.
      dow_cos_index: The index in the observation tensor corresponding to the
        cosine component of the day of the week encoding.
      hod_sin_index: The index in the observation tensor corresponding to the
        sine component of the hour of the day encoding.
      hod_cos_index: The index in the observation tensor corresponding to the
        cosine component of the hour of the day encoding.
      action_normalizers: A dictionary mapping action field identifiers (keys
        should correspond to how normalizers are identified, e.g., by a unique
        string that can be matched with `SetpointName`) to instances of
        `base_normalizer.BaseActionNormalizer`. These are used to convert the
        native setpoint values from the schedule into the normalized action
        values expected by the environment (typically in [-1, 1] range).
      local_start_time: A `pd.Timestamp` (timezone-aware) representing the
        simulation's reference start time. This is used with the hour-of-day
        timedelta to correctly interpret schedule event times relative to
        potential UTC offsets if observations provide time in a different frame.
      name: An optional name for this policy.
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
    # Assuming time features in observation are normalized with mean 0, std 1
    self.norm_mean = 0.0
    self.norm_std = 1.0

    super().__init__(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        policy_state_spec=(), # This policy is stateless
        info_spec=(), # No additional info provided by this policy
        clip=False, # Actions are assumed to be within spec by normalizers
        observation_and_action_constraint_splitter=None,
        name=name,
    )

  def _normalize_actions(
      self, action_map: Dict[Tuple[DeviceType, SetpointName], Optional[SetpointValue]]
  ) -> Dict[Tuple[DeviceType, SetpointName], Optional[types.Float]]:
    """Normalizes native setpoint values to the agent's action space.

    Iterates through the `action_map` (containing native setpoint values)
    and uses the corresponding `action_normalizers` to convert each value
    into the normalized format (typically a float in [-1, 1]) expected by
    the environment's action specification.

    Args:
      action_map: A dictionary mapping `(DeviceType, SetpointName)` tuples to
        their native `SetpointValue` (or `None` if no value was found).

    Returns:
      A dictionary mapping `(DeviceType, SetpointName)` tuples to their
      normalized action values (as floats, or `None` if input was `None`).
    """
    normalized_actions = {}
    for (device, setpoint_name), value in action_map.items():
      if value is None:
        logger.warning("No setpoint value found for %s - %s to normalize.", device, setpoint_name)
        # TODO(user): Decide how to handle missing setpoint values.
        # Option 1: Raise an error.
        # Option 2: Return a default normalized action (e.g., 0.0).
        # Option 3: Allow None to propagate, if environment/agent can handle it.
        # For now, propagating None, assuming downstream handles it or it's an error.
        normalized_actions[(device, setpoint_name)] = None
        continue

      matched_normalizer = None
      # The keys in self.action_normalizers are typically strings like 'device_id_setpoint_name'.
      # We need a robust way to match (DeviceType, SetpointName) to these keys.
      # This simplistic matching might need to be more robust depending on key format.
      # A direct mapping or a more structured key for action_normalizers would be better.
      for normalizer_key_str, normalizer in self.action_normalizers.items():
        # This matching logic assumes normalizer_key_str ends with the setpoint_name
        # and potentially contains device type information implicitly.
        # This is fragile. A better approach would be to have action_normalizers
        # keyed directly by (DeviceType, SetpointName) or a unique DeviceFieldId
        # that is consistently used.
        if normalizer_key_str.endswith(setpoint_name): # Simplified matching
          matched_normalizer = normalizer
          break
      
      if matched_normalizer:
        normalized_actions[(device, setpoint_name)] = matched_normalizer.agent_value(float(value))
      else:
        logger.error("No normalizer found for setpoint: %s. Value %s not normalized.", setpoint_name, value)
        # Decide handling: raise error, or use unnormalized, or default.
        # Propagating original value if not normalizable, though this is likely an error.
        normalized_actions[(device, setpoint_name)] = float(value) # Fallback, likely problematic
        
    return normalized_actions

  def _get_action_map(
      self, time_step: ts.TimeStep
  ) -> Dict[Tuple[DeviceType, SetpointName], Optional[SetpointValue]]:
    """Determines the native setpoint values based on the current time.

    Extracts time features (day of week, hour of day) from the observation,
    selects the appropriate schedule (weekday/weekend), and retrieves the
    active setpoint for each action defined in `self.action_sequence`.

    Args:
      time_step: The current `tf_agents.trajectories.time_step.TimeStep`
        containing the observation.

    Returns:
      A dictionary mapping `(DeviceType, SetpointName)` tuples to their
      determined native `SetpointValue` (or `None` if a value could not be
      determined for a scheduled action).
    """
    observation = time_step.observation # This is a batched observation

    # Assuming observation is structured as (batch_size, num_features)
    # We take the first element of the batch for time processing.
    # If batch_size > 1, this policy might not behave as expected unless
    # all items in batch have same time features or policy is adapted.
    obs_features = observation[0]

    # Denormalize the time signals from the observation
    # These indices (e.g., self.dow_sin_index) must correctly point to the
    # relevant features in the flattened observation tensor.
    dow_sin = (obs_features[self.dow_sin_index] * self.norm_std) + self.norm_mean
    dow_cos = (obs_features[self.dow_cos_index] * self.norm_std) + self.norm_mean
    hod_sin = (obs_features[self.hod_sin_index] * self.norm_std) + self.norm_mean
    hod_cos = (obs_features[self.hod_cos_index] * self.norm_std) + self.norm_mean

    # Convert sine/cosine pairs to day of week (0=Monday, 6=Sunday) and hour of day (0-23)
    dow = to_dow(dow_sin.numpy(), dow_cos.numpy()) # Convert tensor to numpy for util function
    hod = to_hod(hod_sin.numpy(), hod_cos.numpy()) # Convert tensor to numpy

    # Create a Timedelta representing time since midnight, adjusted by local_start_time's UTC offset.
    # This helps align schedule times (defined as offset from midnight local time)
    # with the HoD extracted from potentially UTC-based observations.
    # The assumption is that ScheduleEvent.start_time is local time of day.
    current_time_of_day = pd.Timedelta(hours=hod)
    # If local_start_time is timezone-aware, its utcoffset() can be used.
    # If schedules are defined strictly in local time relative to midnight,
    # and HoD is also local, then utcoffset might not be needed or handled differently.
    # The current usage of local_start_time.utcoffset() seems to adjust for
    # the difference between the observation's time frame (potentially UTC if HoD is from UTC)
    # and the local time frame of the schedules.
    # This needs careful consideration based on how HoD is generated in observations.
    # Assuming HoD is effectively "local hour" for schedule lookup.
    # If HoD is UTC hour, then:
    # local_hod = (hod + local_start_time.utcoffset().total_seconds()/3600) % 24
    # current_time_of_day = pd.Timedelta(hours=local_hod)

    # Select appropriate schedule based on day type
    schedule_to_use = self.weekday_schedule if dow < 5 else self.weekend_schedule # 0-4: Mon-Fri

    # Get active setpoints for each device/setpoint pair in the defined action sequence
    action_values = {}
    for device, setpoint in self.action_sequence:
      action_values[(device, setpoint)] = get_active_setpoint(
          schedule_to_use, device, setpoint, current_time_of_day
      )
    return action_values

  def _action(
      self,
      time_step: ts.TimeStep,
      policy_state: types.NestedTensor,
      seed: Optional[types.Seed] = None  # Add seed arg
  ) -> policy_step.PolicyStep:
    """Computes the action for the given `time_step`.

    This method implements the core logic of the policy:
    1. Determines native setpoint values using `_get_action_map`.
    2. Normalizes these values using `_normalize_actions`.
    3. Constructs the action tensor in the order specified by `self.action_sequence`.
    4. Returns the action as a `PolicyStep`.

    Args:
      time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
      policy_state: A nest of Tensors representing the policy's state. For this
        stateless policy, it's an empty tuple.
      seed: An optional random seed for stochastic policies (unused here).

    Returns:
      A `PolicyStep` named tuple containing the computed action, the policy
      state (empty for this policy), and policy info (empty).
    """
    del seed # This policy is deterministic, seed is not used.

    # Get native action values based on the current time from the schedule
    native_action_map = self._get_action_map(time_step)

    # Normalize these native values into the agent's action space
    normalized_action_map = self._normalize_actions(native_action_map)

    # Construct the action tensor in the order defined by self.action_sequence
    # Handle potential None values from normalization if a setpoint was missing
    action_list = []
    for device, setpoint in self.action_sequence:
        val = normalized_action_map.get((device, setpoint))
        if val is None:
            # Fallback if a normalized value is None: use midpoint of action_spec, or raise error.
            # Assuming action_spec is BoundedTensorSpec for all components.
            # This part needs careful handling of how Nones are treated.
            # For now, logging error and using 0, which might be valid for [-1,1] spec.
            logger.error("Normalized action for (%s, %s) is None. Using 0.0.", device, setpoint)
            action_list.append(0.0)
        else:
            action_list.append(val)

    action_tensor = tf.convert_to_tensor(
        [action_list], # Create a batch of 1
        dtype=self.action_spec.dtype # Ensure dtype matches action_spec
    )
    
    return policy_step.PolicyStep(action=action_tensor, state=policy_state, info=self.info_spec)


def create_baseline_schedule_policy(
    tf_env: tf_py_environment.TFPyEnvironment,
) -> SchedulePolicy:
  """Creates a `SchedulePolicy` configured with a baseline HVAC schedule.

  This function instantiates a `SchedulePolicy` with predefined weekday and
  weekend schedules for Air Conditioning (AC) supply air heating temperature
  and Hot Water System (HWS) supply water temperature. It serves as a default
  rule-based policy, often used for benchmarking or initial data collection
  in smart building control environments.

  The schedules typically involve temperature setbacks during unoccupied hours
  (e.g., nights, weekends) and target temperatures during occupied hours.

  Args:
    tf_env: The `tf_agents.environments.TFPyEnvironment` instance. This is used
      to extract necessary specifications (action spec, time step spec) and
      environment-specific information like observation feature names (for time
      indices) and action normalizers.

  Returns:
    A `SchedulePolicy` instance configured with the baseline schedule.
  """
  # Assumes tf_env wraps a PyEnvironment which has 'pyenv.envs[0]' structure
  # and that this underlying env has 'field_names', 'action_normalizers',
  # and 'current_simulation_timestamp' attributes.
  # This is specific to the smart_control.Environment structure.
  py_env_instance = tf_env.pyenv.envs[0]

  # Extract tensor specifications from the TF environment
  _, action_spec, time_step_spec = spec_utils.get_tensor_specs(tf_env)

  # Determine indices for time features from the environment's observation spec
  # This assumes 'field_names' attribute exists on the PyEnvironment instance
  # and contains the names of observation features in their flattened order.
  hod_cos_index = py_env_instance.field_names.index('hod_cos_000')
  hod_sin_index = py_env_instance.field_names.index('hod_sin_000')
  dow_cos_index = py_env_instance.field_names.index('dow_cos_000')
  dow_sin_index = py_env_instance.field_names.index('dow_sin_000')

  # Define baseline schedules. Temperatures are typically in Kelvin.
  # Weekday schedule:
  # - AC supply air heating setpoint: 292K (18.85°C) from 6 AM to 7 PM, else 285K (11.85°C)
  # - HWS supply water setpoint: 350K (76.85°C) from 6 AM to 7 PM, else 315K (41.85°C)
  weekday_schedule_events: Schedule = [
      ScheduleEvent(start_time=pd.Timedelta(hours=6), device=DeviceType.AC,
                    setpoint_name='supply_air_heating_temperature_setpoint', setpoint_value=292.0),
      ScheduleEvent(start_time=pd.Timedelta(hours=19), device=DeviceType.AC,
                    setpoint_name='supply_air_heating_temperature_setpoint', setpoint_value=285.0),
      ScheduleEvent(start_time=pd.Timedelta(hours=6), device=DeviceType.HWS,
                    setpoint_name='supply_water_setpoint', setpoint_value=350.0),
      ScheduleEvent(start_time=pd.Timedelta(hours=19), device=DeviceType.HWS,
                    setpoint_name='supply_water_setpoint', setpoint_value=315.0),
  ]

  # Weekend/Holiday schedule:
  # - AC supply air heating setpoint: 285K (11.85°C) all day
  # - HWS supply water setpoint: 315K (41.85°C) all day
  # Note: Using pd.Timedelta(hours=0) if the value is constant all day,
  # or define for specific periods if there are changes.
  # For simplicity, defining it at hour 0, assuming get_active_setpoint handles it.
  weekend_holiday_schedule_events: Schedule = [
      ScheduleEvent(start_time=pd.Timedelta(hours=0), device=DeviceType.AC, # Value for the whole day
                    setpoint_name='supply_air_heating_temperature_setpoint', setpoint_value=285.0),
      ScheduleEvent(start_time=pd.Timedelta(hours=0), device=DeviceType.HWS, # Value for the whole day
                    setpoint_name='supply_water_setpoint', setpoint_value=315.0),
  ]

  # The action sequence defines the order of actions in the output tensor.
  # This must match the expectations of the environment and action_spec.
  defined_action_sequence: ActionSequence = [
      (DeviceType.AC, 'supply_air_heating_temperature_setpoint'),
      (DeviceType.HWS, 'supply_water_setpoint'),
  ]

  # Get the environment's local start time, important for aligning schedule's
  # time-of-day logic if observations are relative to a global clock.
  # Ensure it's timezone-aware for correct offset calculations if needed.
  env_local_start_time = py_env_instance.current_simulation_timestamp
  if env_local_start_time.tzinfo is None:
    env_local_start_time = env_local_start_time.tz_localize(DEFAULT_TIME_ZONE)
  else:
    env_local_start_time = env_local_start_time.tz_convert(DEFAULT_TIME_ZONE)


  # Create and return the SchedulePolicy instance
  policy = SchedulePolicy(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      action_sequence=defined_action_sequence,
      weekday_schedule=weekday_schedule_events,
      weekend_schedule=weekend_holiday_schedule_events,
      action_normalizers=py_env_instance.action_normalizers,
      hod_cos_index=hod_cos_index,
      hod_sin_index=hod_sin_index,
      dow_cos_index=dow_cos_index,
      dow_sin_index=dow_sin_index,
      local_start_time=env_local_start_time,
      name="BaselineSchedulePolicy" # Give it a descriptive name
  )

  return policy
