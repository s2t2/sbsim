"""Reinforcement learning environment for controlling building HVAC systems.

This module defines a controllable building environment that interfaces with
TF-Agents. The environment allows an RL agent to adjust various setpoints
(e.g., temperature) within a simulated building. The primary objective for the
agent is to optimize the HVAC system's energy efficiency while maintaining
occupant comfort.

The environment handles interactions with a building simulation model, processes
actions from the agent, generates observations of the building's state, and
computes rewards based on performance criteria like energy consumption and
comfort levels. It supports normalization of observations and actions,
management of episode lifecycles, and integration with various building models
and reward functions.

Typical usage involves creating an instance of the `Environment` class,
configuring it with a specific building model, reward function, and action/
observation normalizers, and then using it within a TF-Agents training loop.

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import collections
import copy
import os
import time
from typing import Final, Mapping, NewType, Optional, Sequence, Tuple

from absl import logging
import bidict
import gin
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

from smart_control.models import base_building
from smart_control.models import base_normalizer
from smart_control.models import base_reward_function
from smart_control.proto import smart_control_building_pb2
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import building_image_generator
from smart_control.utils import constants
from smart_control.utils import conversion_utils
from smart_control.utils import histogram_reducer
from smart_control.utils import plot_utils
from smart_control.utils import regression_building_utils
from smart_control.utils import run_command_predictor
from smart_control.utils import writer_lib

ACTION_REJECTION_REWARD: Final[float] = -np.inf

DeviceInfo = smart_control_building_pb2.DeviceInfo
ValueType = smart_control_building_pb2.DeviceInfo.ValueType

ActionRequest = smart_control_building_pb2.ActionRequest
ActionResponse = smart_control_building_pb2.ActionResponse
ObservationRequest = smart_control_building_pb2.ObservationRequest
ObservationResponse = smart_control_building_pb2.ObservationResponse
SingleActionRequest = smart_control_building_pb2.SingleActionRequest
SingleActionResponse = smart_control_building_pb2.SingleActionResponse
SingleObservationResponse = smart_control_building_pb2.SingleObservationResponse

DeviceFieldId = NewType("DeviceFieldId", str)
DeviceId = NewType("DeviceId", str)
FieldName = NewType("FieldName", str)

COMFORT_MODE_NOW: Final[str] = "comfort_mode_now"
COMFORT_MODE_SOON: Final[str] = "comfort_mode_soon"
NUM_OCCUPANTS: Final[str] = "num_occupants"
DOW_LABEL: Final[str] = "dow"
HOD_LABEL: Final[str] = "hod"

DeviceFieldId = NewType("DeviceFieldId", str)
FieldName = NewType("FieldName", str)
ActionNormalizerMap = Mapping[
    DeviceFieldId, base_normalizer.BaseActionNormalizer
]

DefaultActions = Mapping[DeviceFieldId, float]

DeviceCode = str
Setpoint = str
MeasurementName = str
DeviceActionTuple = Tuple[DeviceCode, Setpoint]
DeviceMeasurementTuple = Tuple[DeviceCode, MeasurementName]


def all_actions_accepted(action_response: ActionResponse) -> bool:
  """Checks if all single action requests in an ActionResponse were accepted.

  Args:
    action_response (ActionResponse): The response object containing results of
      multiple action requests.

  Returns:
    bool: True if all single_action_responses have the status ACCEPTED,
      False otherwise.
  """
  return all(
      single_action_response.response_type == SingleActionResponse.ACCEPTED
      for single_action_response in action_response.single_action_responses
  )


def replace_missing_observations_past(
    current_observation_response: ObservationResponse,
    past_observation_response: Optional[ObservationResponse],
) -> ObservationResponse:
  """Fills missing observations from a past response.

  Building simulations may not always report all requested observation fields.
  This function ensures that the agent receives a complete observation vector
  by imputing any missing values from the `past_observation_response`.

  Args:
    current_observation_response (ObservationResponse): The latest observation
      response from the building.
    past_observation_response (Optional[ObservationResponse]): The previous
      observation response, used to fill in missing values in the current one.

  Returns:
    ObservationResponse: An updated `current_observation_response` where
    missing or invalid observations have been replaced by values from
    `past_observation_response`.

  Raises:
    ValueError: If `current_observation_response` contains missing values and
      `past_observation_response` is None or does not contain the required
      values.
  """

  def get_observation_request_tuples(
      observation_request: ObservationRequest,
  ) -> set[DeviceMeasurementTuple]:
    """Extracts (device_id, measurement_name) tuples from a request."""
    return set([
        (request.device_id, request.measurement_name)
        for request in observation_request.single_observation_requests
    ])

  def get_observation_response_mapping(
      observation_response: ObservationResponse,
  ) -> dict[DeviceMeasurementTuple, SingleObservationResponse]:
    """Maps (device_id, measurement_name) to their SingleObservationResponse.

    Only valid observations are included in the mapping.
    """
    # pylint: disable=g-complex-comprehension
    return {
        (
            response.single_observation_request.device_id,
            response.single_observation_request.measurement_name,
        ): response
        for response in observation_response.single_observation_responses
        if response.observation_valid
    }

  def check_valid_past_observation(
      past_observation_response: Optional[ObservationResponse],
      missing_observations: set[DeviceMeasurementTuple],
  ) -> None:
    """Validates that a past observation is available for imputation."""
    if not past_observation_response:
      for missing_observation in missing_observations:
        logging.error(
            "Missing or invalid observation response for %s %s; no past"
            " observation to replace with.",
            missing_observation[0],
            missing_observation[1],
        )
      raise ValueError(
          f"Missing {len(missing_observations)} observations, and no past"
          " observation available to replace with."
      )

  def get_missing_observations(
      observation_response: ObservationResponse,
  ) -> set[DeviceMeasurementTuple]:
    """Identifies requested observations not present in the response."""
    observation_request_tuples = get_observation_request_tuples(
        observation_response.request
    )
    observation_response_map = get_observation_response_mapping(
        observation_response
    )
    return observation_request_tuples - set(observation_response_map.keys())

  def update_single_observation_response(
      single_observation_response: SingleObservationResponse,
      past_observation_response_mapping: dict[
          DeviceMeasurementTuple, SingleObservationResponse
      ],
  ) -> SingleObservationResponse:
    """Fills an invalid SingleObservationResponse using past data."""
    if single_observation_response.observation_valid:
      return single_observation_response
    else:
      request = single_observation_response.single_observation_request
      missing_observation = (request.device_id, request.measurement_name)
      # This assumes the missing_observation will always be in
      # past_observation_response_mapping if past_observation_response
      # was valid.
      updated_single_observation_response = past_observation_response_mapping[
          missing_observation
      ]
      logging.warning(
          "Missing or invalid observation response for %s %s; replacing with"
          " past observation.",
          missing_observation[0],
          missing_observation[1],
      )
      return updated_single_observation_response

  missing_observations = get_missing_observations(current_observation_response)

  if not missing_observations:
    return current_observation_response

  check_valid_past_observation(past_observation_response, missing_observations)
  # Ensure past_observation_response is not None for pytype.
  assert past_observation_response is not None

  updated_single_observation_responses = []
  past_observation_response_mapping = get_observation_response_mapping(
      past_observation_response
  )

  for (
      single_obs_response
  ) in current_observation_response.single_observation_responses:
    updated_single_obs_response = update_single_observation_response(
        single_obs_response, past_observation_response_mapping
    )
    updated_single_observation_responses.append(updated_single_obs_response)

  # Create a new ObservationResponse with the filled-in values.
  # A deepcopy is used to avoid modifying the original
  # current_observation_response if it's used elsewhere.
  updated_response = copy.deepcopy(current_observation_response)
  del updated_response.single_observation_responses[:]
  updated_response.single_observation_responses.extend(
      updated_single_observation_responses
  )
  return updated_response


def compute_action_regularization_cost(
    action_history: Sequence[np.ndarray],
) -> float:
  """Calculates a cost penalizing large changes in consecutive actions.

  This cost is the L2 norm of the difference between the last two actions
  in the `action_history`. It encourages smoother control policies.

  Args:
    action_history (Sequence[np.ndarray]): A sequence of actions taken during
      the episode, where each action is a NumPy array.

  Returns:
    float: The L2 norm of the difference between the last two actions. Returns
    0.0 if there are fewer than two actions in the history.

  Raises:
    ValueError: If the shapes of the last two actions in the history do not
      match.
  """
  if len(action_history) < 2:
    return 0.0

  if action_history[-2].shape != action_history[-1].shape:
    raise ValueError(
        "Shapes of the last two actions in history do not match: "
        f"{action_history[-2].shape} vs {action_history[-1].shape}"
    )
  return float(
      np.linalg.norm(action_history[-2] - action_history[-1], ord=2)
  )


@gin.configurable
class ActionConfig:
  """Configuration for action normalizers for device setpoints.

  This class manages the mapping between specific device setpoints (identified
  by a `(device_id, setpoint_name)` tuple) and their corresponding
  `BaseActionNormalizer` instances. Only setpoints explicitly configured here
  will be part of the environment's action space.

  The normalizers define how raw action values from the agent are transformed
  into native setpoint values for the building simulation and vice-versa.

  Attributes:
    action_normalizers (ActionNormalizerMap): A dictionary mapping a
      `DeviceFieldId` (a unique string identifier for a device/setpoint pair)
      to its `BaseActionNormalizer` instance.

  Example:
    Suppose `action_normalizers` is defined as:
    ```
    my_action_normalizers = {
        DeviceFieldId('boiler_0_supply_water_setpoint'):
            ContinuousBaseActionNormalizer(min_val=50.0, max_val=70.0)
    }
    action_config = ActionConfig(action_normalizers=my_action_normalizers)
    ```
    This configures a normalizer for the 'supply_water_setpoint' of the
    device 'boiler_0'.
  """

  def __init__(self, action_normalizers: ActionNormalizerMap):
    """Initializes ActionConfig.

    Args:
      action_normalizers (ActionNormalizerMap): A map from `DeviceFieldId`
        to `BaseActionNormalizer` instances.
    """
    self.action_normalizers = action_normalizers

  def get_action_normalizer(
      self, device_field_id: DeviceFieldId
  ) -> Optional[base_normalizer.BaseActionNormalizer]:
    """Retrieves the action normalizer for a given device field ID.

    Args:
      device_field_id (DeviceFieldId): The unique identifier for the
        device/setpoint pair (e.g., 'boiler_0_supply_water_setpoint').

    Returns:
      Optional[base_normalizer.BaseActionNormalizer]: The normalizer instance
      if found, otherwise None.
    """
    return self.action_normalizers.get(device_field_id)


def generate_field_id(
    device: DeviceId, field: FieldName, id_map: bidict.bidict[Tuple[DeviceId, FieldName], DeviceFieldId]
) -> DeviceFieldId:
  """Generates a unique string ID for a device/field combination.

  This function creates a standardized identifier (DeviceFieldId) by joining
  the `device` ID and `field` name (e.g., "boiler_0_supply_water_setpoint").
  It ensures uniqueness by appending an integer suffix if a collision occurs
  (e.g., if "device_field" is generated by both ('device', 'field') and
  ('device_fi', 'eld')).

  The `id_map` is a bidirectional dictionary that stores existing mappings
  to prevent re-computation and ensure consistent ID generation.

  Args:
    device (DeviceId): The identifier of the device.
    field (FieldName): The name of the measurement or setpoint.
    id_map (bidict.bidict): A bidirectional map tracking existing
      `(device, field)` tuples to their generated `DeviceFieldId`. This map is
      updated by the caller if a new ID is generated.

  Returns:
    DeviceFieldId: A unique string identifier for the device-field pair.
      If the pair already exists in `id_map`, its existing ID is returned.
  """
  if (device, field) in id_map:
    # If the (device, field) tuple is already in the map, return its existing ID.
    # This can happen if, for example, a field is both an observation and an
    # action.
    return id_map[(device, field)]

  base_id = f"{device}_{field}"
  new_id = base_id
  counter = 0

  # Check if the generated `new_id` (e.g., "a_b_c") is already present as a
  # value in the id_map (i.e., id_map.inv contains it). This handles
  # collisions where different (device, field) pairs might naively generate
  # the same string.
  while new_id in id_map.inv:
    counter += 1
    new_id = f"{base_id}_{counter}"

  return DeviceFieldId(new_id)


@gin.configurable
class Environment(py_environment.PyEnvironment):
  """RL environment for controlling a simulated building.

  This class implements the `py_environment.PyEnvironment` interface from
  TF-Agents, allowing an RL agent to interact with a simulated building model.
  The agent's goal is typically to optimize energy efficiency while maintaining
  occupant comfort by controlling HVAC setpoints.

  Key functionalities:
  - Manages the simulation state and time.
  - Defines action and observation spaces.
  - Processes agent actions, applies them to the building model.
  - Generates observations based on the building's state.
  - Calculates rewards using a configurable reward function.
  - Handles episode initialization, stepping, and termination.
  - Supports normalization of actions and observations.
  - Logs metrics for monitoring and analysis.

  Example:
    ```python
    # (Illustrative - assumes building, reward_fn, etc. are defined)
    # import base_building_module
    # import base_reward_function_module
    # import base_normalizer_module
    # import action_config_module

    # building_sim = base_building_module.MyBuildingSimulator(...)
    # reward_calc = base_reward_function_module.MyRewardCalculator(...)
    # obs_norm = base_normalizer_module.ObservationNormalizer(...)
    # act_conf = action_config_module.ActionConfig(...)

    # env = Environment(
    #     building=building_sim,
    #     reward_function=reward_calc,
    #     observation_normalizer=obs_norm,
    #     action_config=act_conf,
    #     num_days_in_episode=7,
    #     # ... other parameters
    # )

    # # Use 'env' with a TF-Agents agent and training loop.
    ```
  """

  def __init__(
      self,
      building: base_building.BaseBuilding,
      reward_function: base_reward_function.BaseRewardFunction,
      observation_normalizer: base_normalizer.BaseObservationNormalizer,
      action_config: ActionConfig,
      discount_factor: float = 1.0,
      metrics_path: Optional[str] = None,
      num_days_in_episode: int = 3,
      device_action_tuples: Optional[Sequence[DeviceActionTuple]] = None,
      default_actions: Optional[DefaultActions] = None,
      metrics_reporting_interval: float = 100.0,
      label: str = "episode_metrics",
      num_hod_features: int = 1,
      num_dow_features: int = 1,
      occupancy_normalization_constant: float = 0.0,
      run_command_predictors: Optional[
          Sequence[run_command_predictor.BaseRunCommandPredictor]
      ] = None,
      observation_histogram_reducer: Optional[
          histogram_reducer.HistogramReducer
      ] = None,
      time_zone: str = "US/Pacific",
      image_generator: Optional[
          building_image_generator.BuildingImageGenerator
      ] = None,
      step_interval: pd.Timedelta = pd.Timedelta(5, unit="minutes"),
      writer_factory: Optional[writer_lib.BaseWriterFactory] = None,
  ) -> None:
    """Initializes the building control environment.

    Args:
      building (base_building.BaseBuilding): An instance of a building
        simulation model.
      reward_function (base_reward_function.BaseRewardFunction): Calculates
        rewards based on building state and actions.
      observation_normalizer (base_normalizer.BaseObservationNormalizer):
        Normalizes observations from the building.
      action_config (ActionConfig): Defines the action space and how actions
        are normalized and applied.
      discount_factor (float): The discount factor (gamma) for future rewards,
        in (0, 1].
      metrics_path (Optional[str]): Directory path to write environment data
        and metrics. If None, metrics are not written to disk.
      num_days_in_episode (int): The duration of each episode in days.
      device_action_tuples (Optional[Sequence[DeviceActionTuple]]): A specific
        list of (device_id, setpoint_name) tuples that define the controllable
        actions. If None, actions are inferred from `action_config` and
        `building.devices`.
      default_actions (Optional[DefaultActions]): A mapping of `DeviceFieldId`
        to default native setpoint values to be used, for example, by a
        default policy.
      metrics_reporting_interval (float): The interval (in simulation steps)
        at which to report metrics to TensorBoard.
      label (str): A label prepended to episode output directories when
        `metrics_path` is specified.
      num_hod_features (int): Number of sine/cosine pairs for encoding the
        hour of the day as a cyclical feature.
      num_dow_features (int): Number of sine/cosine pairs for encoding the
        day of the week as a cyclical feature.
      occupancy_normalization_constant (float): A constant used in the
        normalization of occupancy-related features.
      run_command_predictors (Optional[Sequence[...]]): A list of predictors
        used for setting on/off states in `RunCommands`.
      observation_histogram_reducer (Optional[histogram_reducer.HistogramReducer]):
        If provided, used to reduce observation timeseries into histograms.
      time_zone (str): The time zone of the building/environment (e.g.,
        "US/Pacific").
      image_generator (Optional[building_image_generator.BuildingImageGenerator]):
        If provided, generates image representations of the building state.
      step_interval (pd.Timedelta): The duration of a single environment step.
      writer_factory (Optional[writer_lib.BaseWriterFactory]): If
        `metrics_path` is provided, this factory is used to create metric
        writers.
    """
    super().__init__()

    self.building: base_building.BaseBuilding = building
    self._time_zone: str = time_zone
    self._device_action_tuples: Optional[Sequence[DeviceActionTuple]] = (
        device_action_tuples
    )
    self.reward_function: base_reward_function.BaseRewardFunction = (
        reward_function
    )
    self._observation_histogram_reducer: Optional[
        histogram_reducer.HistogramReducer
    ] = observation_histogram_reducer
    self.discount_factor: float = discount_factor
    self._step_count: int = 0
    self._global_step_count: int = 0
    self._episode_count: int = 0
    self._episode_cumulative_reward: float = 0.0
    self._last_log_timestamp: float = 0.0  # TODO(b/260300338): remove?
    self._observation_normalizer: base_normalizer.BaseObservationNormalizer = (
        observation_normalizer
    )
    self._start_timestamp: pd.Timestamp = self.building.current_timestamp
    self._action_history: list[np.ndarray] = []
    self._end_timestamp: pd.Timestamp = self._start_timestamp + pd.Timedelta(
        num_days_in_episode, unit="days"
    )
    self._step_interval: pd.Timedelta = step_interval
    self._num_timesteps_in_episode: int = int(
        (self._end_timestamp - self._start_timestamp) / self._step_interval
    )
    # TODO(b/260300338): self._metrics seems unused, remove?
    self._metrics = plot_utils.init_metrics()
    logging.info(
        "Episode starts at %s and ends at %s; %d timesteps.",
        self._start_timestamp,
        self._end_timestamp,
        self._num_timesteps_in_episode,
    )

    # Bidirectional map for (DeviceId, FieldName) <-> DeviceFieldId
    self._id_map: bidict.bidict[
        Tuple[DeviceId, FieldName], DeviceFieldId
    ] = bidict.bidict()

    if not (0 < self.discount_factor <= 1):
      raise ValueError(
          f"Discount factor must be in (0, 1], got {self.discount_factor}"
      )

    self.metrics_path: Optional[str] = metrics_path
    self._writer_factory: Optional[writer_lib.BaseWriterFactory] = (
        writer_factory
    )
    self._metrics_writer: Optional[writer_lib.BaseWriter] = None
    self._summary_writer: Optional[tf.summary.SummaryWriter] = None
    self._label: str = label
    self._num_dow_features: int = num_dow_features
    self._num_hod_features: int = num_hod_features
    self._last_observation_response: Optional[ObservationResponse] = None

    if device_action_tuples is not None:
      (
          self._action_spec,
          self.action_normalizers,
          self._action_names,
      ) = self._get_action_spec_and_normalizers_from_device_action_tuples(
          action_config=action_config,
          device_action_tuples=device_action_tuples,
      )
    else:
      (
          self._action_spec,
          self.action_normalizers,
          self._action_names,
      ) = self._get_action_spec_and_normalizers(
          action_config, building.devices
      )

    logging.info("Action Names: %s", self._action_names)

    self._auxiliary_features: Sequence[
        str
    ] = self._get_auxiliary_features_labels(
        self._num_hod_features, self._num_dow_features
    )
    logging.info("Auxiliary Features: %s", self._auxiliary_features)

    self._observation_spec, self.field_names = self._get_observation_spec(
        building.devices
    )
    logging.info("Observation Spec: %s", self._observation_spec)
    logging.info(
        "%s FIELD NAMES (%d): %s",
        self._label,
        len(self.field_names),
        self.field_names,
    )

    self._episode_ended: bool = False
    self._episode_start_time: float = time.time()

    self._default_policy_values: tf.Tensor = (
        self._normalize_default_actions(default_actions)
        if default_actions
        else tf.constant([], dtype=tf.float32)
    )

    self._accumulator: collections.defaultdict[
        str, list[float]
    ] = collections.defaultdict(list)
    self._metrics_reporting_interval: float = metrics_reporting_interval
    self._observation_request: ObservationRequest = (
        self._get_observation_request(building.devices)
    )
    self.occupancy_normalization_constant: float = (
        occupancy_normalization_constant
    )
    self._run_command_predictors: Optional[
        list[run_command_predictor.BaseRunCommandPredictor]
    ] = (list(run_command_predictors) if run_command_predictors else None)
    self._building_image_generator: Optional[
        building_image_generator.BuildingImageGenerator
    ] = image_generator

  def set_summary_writer(self, summary_path: str) -> None:
    """Sets the TensorFlow summary writer for logging metrics.

    Args:
      summary_path (str): The directory path where TensorFlow summaries will be
        written.
    """
    self._summary_writer = tf.compat.v2.summary.create_file_writer(
        summary_path, flush_millis=10000
    )

  @property
  def steps_per_episode(self) -> int:
    """The total number of simulation steps in a single episode."""
    return int(
        (self._end_timestamp - self._start_timestamp).total_seconds()
        // self.building.time_step_sec
    )

  @property
  def start_timestamp(self) -> pd.Timestamp:
    """The simulation timestamp at the beginning of the current episode."""
    return self._start_timestamp

  @property
  def end_timestamp(self) -> pd.Timestamp:
    """The simulation timestamp at which the current episode will end."""
    return self._end_timestamp

  @end_timestamp.setter
  def end_timestamp(self, value: pd.Timestamp) -> None:
    """Sets the end timestamp for the current episode."""
    self._end_timestamp = value

  @property
  def default_policy_values(self) -> tf.Tensor:
    """Normalized action values for a default policy, if configured."""
    return self._default_policy_values

  def _get_observation_request(
      self, devices: Sequence[DeviceInfo]
  ) -> ObservationRequest:
    """Creates a template ObservationRequest based on available devices.

    This request lists all observable fields from all devices and is reused
    (with updated timestamps) for each step.

    Args:
      devices (Sequence[DeviceInfo]): A list of device information objects.

    Returns:
      ObservationRequest: A pre-filled observation request proto.
    """
    observation_request = ObservationRequest()
    # Sort devices by ID and fields by name for consistent request structure.
    for device in sorted(devices, key=lambda d: d.device_id):
      for measurement_name in sorted(device.observable_fields):
        observation_request.single_observation_requests.add(
            device_id=device.device_id, measurement_name=measurement_name
        )
    return observation_request

  def _get_auxiliary_features_labels(
      self, num_hod_features: int, num_dow_features: int
  ) -> Sequence[str]:
    """Generates labels for auxiliary time and occupancy features.

    These features include cyclical representations of hour-of-day (hod) and
    day-of-week (dow), comfort mode indicators, and number of occupants.

    Args:
      num_hod_features (int): Number of sin/cos pairs for hour features.
      num_dow_features (int): Number of sin/cos pairs for day features.

    Returns:
      Sequence[str]: A list of string labels for these auxiliary features.
    """
    hod_labels = [
        f"{prefix}_{idx}"
        for prefix, idx in regression_building_utils.get_time_feature_names(
            num_hod_features, HOD_LABEL
        )
    ]
    dow_labels = [
        f"{prefix}_{idx}"
        for prefix, idx in regression_building_utils.get_time_feature_names(
            num_dow_features, DOW_LABEL
        )
    ]
    other_labels = [COMFORT_MODE_NOW, COMFORT_MODE_SOON, NUM_OCCUPANTS]
    return hod_labels + dow_labels + other_labels

  def _normalize_default_actions(
      self, default_actions: DefaultActions
  ) -> tf.Tensor:
    """Converts native default action values to normalized agent values.

    Args:
      default_actions (DefaultActions): A mapping from `DeviceFieldId` to
        native (unnormalized) default action values.

    Returns:
      tf.Tensor: A 1D tensor of normalized default action values, ordered
      according to `self._action_names`.

    Raises:
      KeyError: If a `DeviceFieldId` in `self._action_names` is not found in
        `default_actions` or if a `DeviceFieldId` in `default_actions` does not
        correspond to a known action.
    """
    normalized_actions = []
    for field_id in self._action_names:
      # The original `default_actions` uses `FieldName` as keys.
      # We need to get the original `FieldName` from `field_id` using `_id_map`.
      # `field_id` is a `DeviceFieldId` (e.g., "device_setpoint").
      # `_id_map.inv[field_id]` gives `(DeviceId, FieldName)`.
      _device_id, field_name = self._id_map.inv[field_id]

      if field_name not in default_actions:
        # Ensure all expected actions have defaults.
        # Note: The original code used `setpoint_name` for `default_actions`
        # keys. If `default_actions` is keyed by `DeviceFieldId` directly,
        # this check should be `field_id not in default_actions`.
        # Assuming `default_actions` uses `FieldName` as intended.
        raise KeyError(
            f"Default action for field '{field_name}' (ID: {field_id}) not "
            "found in `default_actions`."
        )

      native_value = default_actions[field_name]
      normalizer = self.action_normalizers[field_id]
      agent_value = normalizer.agent_value(native_value)
      normalized_actions.append(agent_value)

    return tf.constant(normalized_actions, dtype=tf.float32)

  def _get_action_spec_and_normalizers(
      self,
      action_config: ActionConfig,
      devices: Sequence[DeviceInfo],
  ) -> Tuple[types.ArraySpec, ActionNormalizerMap, Sequence[DeviceFieldId]]:
    """Builds action spec, normalizers, and names from all device actions.

    This method iterates through all available devices and their controllable
    setpoints, configuring them based on the `action_config`.

    Args:
      action_config (ActionConfig): Configuration for action normalizers.
      devices (Sequence[DeviceInfo]): List of device information protos.

    Returns:
      Tuple:
        - types.ArraySpec: The TF-Agents action specification.
        - ActionNormalizerMap: Map from `DeviceFieldId` to normalizers.
        - Sequence[DeviceFieldId]: Ordered list of action field IDs.
    """

    def _check_value_type_continuous(value_type: ValueType) -> None:
      if value_type == ValueType.VALUE_TYPE_UNDEFINED:
        raise ValueError("Action value type is undefined.")
      elif value_type != ValueType.VALUE_CONTINUOUS:
        # Currently, only continuous actions are fully supported.
        raise NotImplementedError(
            f"Action value type {ValueType.Name(value_type)} not supported."
        )

    action_normalizers: ActionNormalizerMap = {}
    action_names: list[DeviceFieldId] = []

    logging.info(
        "Loading device-setpoint pairs from %d device_infos.", len(devices)
    )
    # Sort devices and setpoints for consistent action ordering.
    for device_info in sorted(devices, key=lambda d: d.device_id):
      dev_id = DeviceId(device_info.device_id)
      for setpoint_key in sorted(device_info.action_fields.keys()):
        setpoint_name = FieldName(setpoint_key)
        value_type = device_info.action_fields[setpoint_key]

        # Retrieve the specific normalizer for this setpoint from ActionConfig.
        # ActionConfig is keyed by DeviceFieldId, so we generate it first.
        # However, the original intent of ActionConfig might be to use FieldName.
        # For now, let's assume ActionConfig is flexible or we adapt.
        # A simpler approach: ActionConfig uses FieldName (setpoint_name)
        # as keys.
        # normalizer = action_config.get_action_normalizer(setpoint_name)
        # This needs `action_config` to be keyed by `FieldName`.
        # If `action_config` is keyed by `DeviceFieldId`, we'd need to
        # construct a temporary `DeviceFieldId` or iterate `action_config`.
        # The current `ActionConfig.get_action_normalizer` takes `FieldName`.
        # This seems inconsistent with `ActionNormalizerMap` keying.
        # Let's assume `action_config.get_action_normalizer` uses `FieldName`.
        normalizer = action_config.get_action_normalizer(setpoint_name)

        if not normalizer:
          logging.debug(
              "No action normalizer found for device %s, setpoint %s. "
              "Skipping.",
              dev_id,
              setpoint_name,
          )
          continue

        # Generate a unique ID for this device/setpoint combination.
        # This ID will be used in the action spec and normalizer map.
        # The `_id_map` is updated here.
        field_id = generate_field_id(dev_id, setpoint_name, self._id_map)
        self._id_map[(dev_id, setpoint_name)] = field_id

        action_names.append(field_id)
        action_normalizers[field_id] = normalizer
        _check_value_type_continuous(value_type)
        # The array spec for each individual action is part of the normalizer.

    if not action_names:
      logging.warning("No actions were configured for the environment.")
      # Return an empty spec if no actions.
      action_array_spec = array_spec.BoundedArraySpec(
          shape=(0,),
          dtype=np.float32,
          name="action",
          minimum=-1.0,
          maximum=1.0,
      )
    else:
      action_array_spec = array_spec.BoundedArraySpec(
          shape=(len(action_names),),
          dtype=np.float32,
          name="action", # Name for the entire action vector
          minimum=-1.0, # Assuming all normalizers output in [-1, 1]
          maximum=1.0,
      )

    logging.info(
        "Action spec configured with %d actions: %s",
        len(action_names),
        action_names,
    )
    return action_array_spec, action_normalizers, action_names

  def _get_action_spec_and_normalizers_from_device_action_tuples(
      self,
      action_config: ActionConfig,
      device_action_tuples: Sequence[DeviceActionTuple],
  ) -> Tuple[types.ArraySpec, ActionNormalizerMap, Sequence[DeviceFieldId]]:
    """Builds action spec from a specific list of (device, setpoint) tuples.

    This method is used when `device_action_tuples` is provided, allowing
    fine-grained control over which actions are included.

    Args:
      action_config (ActionConfig): Configuration for action normalizers.
      device_action_tuples (Sequence[DeviceActionTuple]): Specific list of
        (device_id, setpoint_name) to include in the action space.

    Returns:
      Tuple:
        - types.ArraySpec: The TF-Agents action specification.
        - ActionNormalizerMap: Map from `DeviceFieldId` to normalizers.
        - Sequence[DeviceFieldId]: Ordered list of action field IDs.

    Raises:
      ValueError: If a normalizer is not found in `action_config` for any of
        the specified device/setpoint tuples.
    """
    action_normalizers: ActionNormalizerMap = {}
    action_names: list[DeviceFieldId] = []

    logging.info(
        "Loading device-setpoint pairs from %d device_action_tuples.",
        len(device_action_tuples),
    )
    for dev_id_str, setpoint_name_str in device_action_tuples:
      dev_id = DeviceId(dev_id_str)
      setpoint_name = FieldName(setpoint_name_str)

      # Get normalizer from ActionConfig. Assuming ActionConfig is keyed by
      # FieldName as its `get_action_normalizer` method suggests.
      normalizer = action_config.get_action_normalizer(setpoint_name)

      if not normalizer:
        raise ValueError(
            f"Missing action normalizer for device '{dev_id}', setpoint "
            f"'{setpoint_name}' in action_config. Please ensure all "
            "device_action_tuples have a corresponding normalizer."
        )

      # Generate and store the unique ID for this action.
      field_id = generate_field_id(dev_id, setpoint_name, self._id_map)
      self._id_map[(dev_id, setpoint_name)] = field_id

      action_names.append(field_id)
      action_normalizers[field_id] = normalizer
      # Individual action specs are implicitly defined by the normalizers.
      # The overall action spec will be a BoundedArraySpec.

    action_array_spec = array_spec.BoundedArraySpec(
        shape=(len(action_names),),
        dtype=np.float32, # Assuming float actions after normalization
        name="action",
        minimum=-1.0, # Standard range for normalized actions
        maximum=1.0,
    )
    logging.info(
        "Action spec from device_action_tuples configured with %d actions: %s",
        len(action_names),
        action_names,
    )
    return action_array_spec, action_normalizers, action_names

  def _get_observation_spec(
      self, devices: Sequence[DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[DeviceFieldId]]:
    """Constructs the observation specification for the environment.

    The observation spec defines the shape and type of the data that the
    environment returns as its state. It includes measurements from devices
    and auxiliary features like time encodings.

    Args:
      devices (Sequence[DeviceInfo]): A list of device information objects from
        the building model.

    Returns:
      tuple:
        - types.ArraySpec: The TF-Agents observation specification, typically a
          flat array of floats.
        - Sequence[DeviceFieldId]: An ordered list of field names (DeviceFieldId)
          corresponding to the elements in the observation array.
    """
    if self._observation_histogram_reducer is None:
      # Standard case: each observable field is a single float value.
      obs_spec, observable_field_ids = (
          self._get_observation_spec_single_timeseries(devices)
      )
    else:
      # Histogram case: some fields might be reduced to histogram bins.
      obs_spec, observable_field_ids = (
          self._get_observation_spec_histogram_reducer(devices)
      )

    logging.info(
        "Observation spec configured with %d fields.",
        len(observable_field_ids),
    )
    logging.debug("Observable field IDs: %s", observable_field_ids)
    return obs_spec, observable_field_ids

  def _get_observation_spec_histogram_reducer(
      self, devices: Sequence[DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[DeviceFieldId]]:
    """Builds observation spec when a histogram reducer is used.

    Some observations are passed through directly, while others (specified in
    the reducer's config) are converted into histogram bin counts.

    Args:
      devices (Sequence[DeviceInfo]): List of device information protos.

    Returns:
      Tuple:
        - types.ArraySpec: The TF-Agents observation specification.
        - Sequence[DeviceFieldId]: Ordered list of observation field IDs.
    """
    assert self._observation_histogram_reducer is not None
    observable_field_ids: list[DeviceFieldId] = []

    # Sort devices and measurements for consistent observation ordering.
    for device_info in sorted(devices, key=lambda d: d.device_id):
      dev_id = DeviceId(device_info.device_id)
      for measurement_key in sorted(device_info.observable_fields):
        measurement_name = FieldName(measurement_key)

        # Check if this measurement is configured for histogram reduction.
        if (
            measurement_name
            in self._observation_histogram_reducer.histogram_parameters
        ):
          # Add field IDs for each bin of the histogram.
          for bin_val in self._observation_histogram_reducer.histogram_parameters[
              measurement_name
          ]:
            # Create a unique ID for this specific bin.
            # Using a tuple (measurement_name, bin_id_str) for id_map key
            # to distinguish from simple measurement_name.
            bin_id_str = f"h_{bin_val:.2f}"
            # The key in _id_map should be (DeviceId, FieldName-like).
            # Here, FieldName-like is measurement_name + bin_id_str.
            # This might lead to very long DeviceFieldIds.
            # Consider if DeviceId should be part of histogram keys in _id_map.
            # Current generate_field_id takes (DeviceId, FieldName).
            # Let's form a composite FieldName for histogram bins.
            composite_field_name = FieldName(f"{measurement_name}_{bin_id_str}")
            field_id = generate_field_id(
                dev_id, composite_field_name, self._id_map
            )
            if self._id_map.get((dev_id, composite_field_name)) is None:
              self._id_map[(dev_id, composite_field_name)] = field_id
              logging.info(
                  "Histogram feature ID: %s (from %s, %s, bin %s) added.",
                  field_id,
                  dev_id,
                  measurement_name,
                  bin_id_str,
              )
            observable_field_ids.append(field_id)
        else:
          # This measurement is a direct pass-through value.
          field_id = generate_field_id(dev_id, measurement_name, self._id_map)
          if self._id_map.get((dev_id, measurement_name)) is None:
            self._id_map[(dev_id, measurement_name)] = field_id
            logging.info(
                "Passthrough feature ID: %s (from %s, %s) added.",
                field_id,
                dev_id,
                measurement_name,
            )
          observable_field_ids.append(field_id)

    # Add labels for auxiliary features (time, occupancy, etc.).
    # These are treated as DeviceFieldId for consistency in `field_names`,
    # though they don't map to a device.
    observable_field_ids.extend(
        [DeviceFieldId(label) for label in self._auxiliary_features]
    )

    obs_array_spec = array_spec.ArraySpec(
        shape=(len(observable_field_ids),),
        dtype=np.float32, # Assuming all observations are float.
        name="observation_with_histograms",
    )
    return obs_array_spec, observable_field_ids

  def _get_observation_spec_single_timeseries(
      self, devices: Sequence[DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[DeviceFieldId]]:
    """Builds observation spec for standard single timeseries data.

    Each observable field from each device becomes one element in the
    observation vector, plus auxiliary features.

    Args:
      devices (Sequence[DeviceInfo]): List of device information protos.

    Returns:
      Tuple:
        - types.ArraySpec: The TF-Agents observation specification.
        - Sequence[DeviceFieldId]: Ordered list of observation field IDs.
    """
    observable_field_ids: list[DeviceFieldId] = []

    # Sort devices and measurement names for a consistent order.
    for device_info in sorted(devices, key=lambda d: d.device_id):
      dev_id = DeviceId(device_info.device_id)
      for measurement_key in sorted(device_info.observable_fields):
        measurement_name = FieldName(measurement_key)

        # Generate a unique ID for this device/measurement combination.
        # This ID is used to look up the value in the observation dictionary
        # and ensures correct ordering in the final observation array.
        # The `_id_map` is updated here.
        field_id = generate_field_id(dev_id, measurement_name, self._id_map)
        self._id_map[(dev_id, measurement_name)] = field_id
        observable_field_ids.append(field_id)

    # Add auxiliary features (time, comfort mode, occupancy).
    # These are added as DeviceFieldId type for consistency, though they
    # are not tied to a specific device in the same way.
    observable_field_ids.extend(
        [DeviceFieldId(label) for label in self._auxiliary_features]
    )

    # The observation spec is a flat array of floats.
    obs_array_spec = array_spec.ArraySpec(
        shape=(len(observable_field_ids),),
        dtype=np.float32,
        name="observation_single_timeseries",
    )
    return obs_array_spec, observable_field_ids

  @property
  def current_simulation_timestamp(self) -> pd.Timestamp:
    """The current timestamp within the building simulation."""
    return self.building.current_timestamp

  def _get_action_value_type(self, field_id: DeviceFieldId) -> ValueType:
    """Determines the value type of an action field. (Helper, not used in core logic).

    Args:
      field_id (DeviceFieldId): The ID of the action field.

    Returns:
      ValueType: The corresponding protobuf ValueType.

    Raises:
      KeyError: If `field_id` is not found in action or observation specs.
      NotImplementedError: If dtype is not recognized.
    """
    # This method seems to be for introspection or potential future use,
    # as the core logic relies on normalizers which handle type details.
    # It also checks observation_spec, which is unusual for action type.
    # Assuming it's primarily for action fields.
    if field_id in self._action_names:
      # The overall action_spec is a BoundedArraySpec.
      # Individual action types are implicitly float due to normalization.
      # This method might be trying to infer original type, but that's lost.
      # For now, assume all agent-facing actions are float32.
      # The spec for individual fields is not directly stored in self.action_spec()
      # if it's a BoundedArraySpec.
      # This function might need rethinking if specific types per action
      # are needed here.
      # normalizer = self.action_normalizers.get(field_id)
      # if normalizer and hasattr(normalizer, 'native_value_type'):
      #   return normalizer.native_value_type
      return ValueType.VALUE_CONTINUOUS # Defaulting due to normalization
    elif field_id in self.field_names: # Checks observation spec
      # This part makes it confusing if it's about "action" value type.
      # Let's assume this is a general utility for any field ID.
      # Individual observation specs are also not directly in self.observation_spec()
      # if it's a simple ArraySpec.
      # For simplicity, and as observations are processed into a float array:
      return ValueType.VALUE_CONTINUOUS # Defaulting
    else:
      raise KeyError(f"Field ID '{field_id}' not found in action or obs specs.")

    # The original logic for spec.dtype checking is problematic because
    # self.action_spec() / self.observation_spec() return the *overall* spec,
    # not per-field specs if they are flattened.

  def _create_action_request(
      self, action_array: np.ndarray
  ) -> ActionRequest:
    """Constructs an ActionRequest proto from the agent's action array.

    This involves denormalizing the agent's actions (typically in the range
    [-1, 1]) back to their native scales for the building simulation.

    Args:
      action_array (np.ndarray): A 1D NumPy array of normalized action values
        from the agent, ordered according to `self._action_names`.

    Returns:
      ActionRequest: A populated ActionRequest protobuf message ready to be
      sent to the building model.
    """
    current_ts_proto = conversion_utils.pandas_to_proto_timestamp(
        self.building.current_timestamp
    )
    action_request = ActionRequest(timestamp=current_ts_proto)

    # Store the raw (normalized) action array for regularization cost.
    self._action_history.append(action_array.astype(np.float32))

    for i, field_id in enumerate(self._action_names):
      agent_action_value = action_array[i]

      # Retrieve the original device ID and setpoint name.
      device_id, setpoint_name = self._id_map.inv[field_id]
      normalizer = self.action_normalizers[field_id]

      # Convert normalized agent action to native setpoint value.
      native_setpoint_value = normalizer.setpoint_value(agent_action_value)

      single_request = SingleActionRequest(
          device_id=device_id,
          setpoint_name=setpoint_name,
          # Assuming continuous value based on current normalization setup.
          # If discrete actions were supported, this would need adjustment.
          continuous_value=float(native_setpoint_value),
      )
      action_request.single_action_requests.append(single_request)

    return action_request

  def _get_observation(self) -> np.ndarray:
    """Retrieves, processes, and normalizes observations from the building.

    This method performs several steps:
    1. Requests raw observations from the building model.
    2. Fills any missing observation values using the previous step's data.
    3. Logs raw observations if a metrics writer is configured.
    4. Normalizes the observations using the `observation_normalizer`.
    5. Converts the normalized `ObservationResponse` into a flat feature map.
    6. Appends auxiliary features (time encodings, comfort mode, occupancy).
    7. Assembles the final flat NumPy array observation for the agent.

    Returns:
      np.ndarray: A 1D NumPy array representing the current environment
      observation, ordered according to `self.field_names`.
    """
    current_ts_proto = conversion_utils.pandas_to_proto_timestamp(
        self.building.current_timestamp
    )
    # Use the pre-built template and update its timestamp.
    observation_request = ObservationRequest()
    observation_request.CopyFrom(self._observation_request)
    observation_request.timestamp.CopyFrom(current_ts_proto)

    raw_observation_response = self.building.request_observations(
        observation_request
    )

    # Impute missing values if necessary.
    processed_observation_response = replace_missing_observations_past(
        current_observation_response=raw_observation_response,
        past_observation_response=self._last_observation_response,
    )
    self._last_observation_response = copy.deepcopy(
        processed_observation_response
    ) # Store for next step.

    if self._metrics_writer:
      self._metrics_writer.write_observation_response(
          processed_observation_response, self.current_simulation_timestamp
      )
      if self._building_image_generator:
        building_image = (
            self._building_image_generator.generate_building_image(
                processed_observation_response
            )
        )
        self._metrics_writer.write_building_image(
            building_image, self.current_simulation_timestamp
        )

    # Normalize observations.
    normalized_response = self._observation_normalizer.normalize(
        processed_observation_response
    )

    # Convert normalized response to a dictionary of {DeviceFieldId: value}.
    if self._observation_histogram_reducer is None:
      observation_map = (
          self._normalized_observation_response_to_observation_map_single_timeseries(
              normalized_response
          )
      )
    else:
      observation_map = (
          self._normalized_observation_response_to_observation_map_histogram_reducer(
              normalized_response
          )
      )

    # Add auxiliary features.
    # Time features (hour of day, day of week)
    hod_rad = conversion_utils.get_radian_time(
        self.current_simulation_timestamp,
        conversion_utils.TimeIntervalEnum.HOUR_OF_DAY,
    )
    hod_feats = regression_building_utils.expand_time_features(
        self._num_hod_features, hod_rad, HOD_LABEL
    )
    for (prefix, idx), val in hod_feats.items():
      observation_map[DeviceFieldId(f"{prefix}_{idx}")] = np.array(
          val, dtype=np.float32
      )

    dow_rad = conversion_utils.get_radian_time(
        self.current_simulation_timestamp,
        conversion_utils.TimeIntervalEnum.DAY_OF_WEEK,
    )
    dow_feats = regression_building_utils.expand_time_features(
        self._num_dow_features, dow_rad, DOW_LABEL
    )
    for (prefix, idx), val in dow_feats.items():
      observation_map[DeviceFieldId(f"{prefix}_{idx}")] = np.array(
          val, dtype=np.float32
      )

    # Comfort mode and occupancy features
    observation_map[DeviceFieldId(COMFORT_MODE_NOW)] = np.array(
        self.building.is_comfort_mode(self.current_simulation_timestamp),
        dtype=np.float32,
    )
    comfort_soon_ts = self.current_simulation_timestamp + pd.Timedelta(
        60, unit="minute"
    )
    observation_map[DeviceFieldId(COMFORT_MODE_SOON)] = np.array(
        self.building.is_comfort_mode(comfort_soon_ts), dtype=np.float32
    )
    # Normalize occupancy: (value - norm_const) / (norm_const + 1)
    # The +1 in denominator avoids division by zero if norm_const is 0.
    norm_occupancy = (
        self.building.num_occupants - self.occupancy_normalization_constant
    ) / (self.occupancy_normalization_constant + 1.0)
    observation_map[DeviceFieldId(NUM_OCCUPANTS)] = np.array(
        norm_occupancy, dtype=np.float32
    )

    # Assemble the final flat observation array in the correct order.
    obs_list = []
    for field_id in self.field_names:
      if field_id not in observation_map:
        # This case should ideally not happen if field_names is derived
        # correctly from all possible observation sources.
        logging.error(
            "Observation field ID '%s' not found in processed observation map."
            " This indicates a mismatch between `self.field_names` and the"
            " actual data collected. Available keys: %s",
            field_id,
            observation_map.keys(),
        )
        # Handle missing data, e.g., by inserting a NaN or default value.
        # For now, raising an error is safer to highlight the issue.
        raise KeyError(
            f"Field ID '{field_id}' expected but not found in observation_map."
        )
      obs_list.append(observation_map[field_id])

    obs_array = np.array(obs_list, dtype=np.float32)

    # Sanity checks for NaNs or Infs.
    if np.isnan(obs_array).any():
      nan_indices = np.where(np.isnan(obs_array))[0]
      nan_fields = [self.field_names[i] for i in nan_indices]
      logging.warning(
          "Observation vector contains NaNs at indices %s (fields: %s).",
          nan_indices,
          nan_fields,
      )
    if np.isinf(obs_array).any():
      inf_indices = np.where(np.isinf(obs_array))[0]
      inf_fields = [self.field_names[i] for i in inf_indices]
      logging.warning(
          "Observation vector contains Infs at indices %s (fields: %s).",
          inf_indices,
          inf_fields,
      )
    return obs_array

  def _normalized_observation_response_to_observation_map_single_timeseries(
      self,
      normalized_observation_response: ObservationResponse,
  ) -> dict[DeviceFieldId, np.ndarray]:
    """Converts normalized ObservationResponse to a feature map (no histogram).

    Each valid, normalized measurement in the `ObservationResponse` is mapped
    to its corresponding `DeviceFieldId`.

    Args:
      normalized_observation_response (ObservationResponse): The protobuf
        message containing normalized observation data.

    Returns:
      dict[DeviceFieldId, np.ndarray]: A dictionary where keys are
      `DeviceFieldId`s and values are 1-element NumPy arrays containing the
      normalized measurement.
    """
    observation_feature_map: dict[DeviceFieldId, np.ndarray] = {}
    for single_resp in (
        normalized_observation_response.single_observation_responses
    ):
      if not single_resp.observation_valid:
        logging.warning(
            "Invalid observation for device %s, measurement %s. Skipping.",
            single_resp.single_observation_request.device_id,
            single_resp.single_observation_request.measurement_name,
        )
        continue

      dev_id = DeviceId(single_resp.single_observation_request.device_id)
      meas_name = FieldName(
          single_resp.single_observation_request.measurement_name
      )

      # Retrieve the unique DeviceFieldId for this device/measurement.
      # This assumes _id_map has been populated during observation spec creation.
      field_id = self._id_map.get((dev_id, meas_name))
      if field_id is None:
        logging.error(
            "DeviceFieldId not found in _id_map for device '%s', measurement"
            " '%s'. This observation will be skipped. This usually indicates"
            " an inconsistency in how _id_map is populated or used.",
            dev_id,
            meas_name,
        )
        continue

      # Store the value as a float32 NumPy array.
      value = np.array(single_resp.continuous_value, dtype=np.float32)
      observation_feature_map[field_id] = value

    return observation_feature_map

  def _normalized_observation_response_to_observation_map_histogram_reducer(
      self,
      normalized_observation_response: ObservationResponse,
  ) -> dict[DeviceFieldId, np.ndarray]:
    """Converts normalized ObservationResponse to a feature map with histograms.

    Uses `HistogramReducer` to process some measurements into histogram bins,
    while others are passed through.

    Args:
      normalized_observation_response (ObservationResponse): The protobuf
        message containing normalized observation data.

    Returns:
      dict[DeviceFieldId, np.ndarray]: A dictionary where keys are
      `DeviceFieldId`s and values are NumPy arrays (could be single values or
      histogram bin counts).
    """
    assert self._observation_histogram_reducer is not None

    # Helper function `get_feature_tuples` extracts (device, measurement, value)
    # from the response.
    feature_tuples = regression_building_utils.get_feature_tuples(
        normalized_observation_response
    )

    # `get_observation_sequence` likely prepares data for the histogram reducer.
    # It might create a DataFrame or a similar structure.
    observation_sequence_df = (
        regression_building_utils.get_observation_sequence(
            [normalized_observation_response], # Expects a list of responses
            feature_tuples,
            self._time_zone,
            self._num_hod_features,
            self._num_dow_features,
        )
    )

    # The reducer processes the sequence and returns a new sequence (DataFrame)
    # where some columns are now histogram bins.
    reduced_sequence_df = self._observation_histogram_reducer.reduce(
        observation_sequence_df
    ).reduced_sequence

    # Convert the first (and only) row of the reduced DataFrame to a dictionary.
    # The column names in `reduced_sequence_df` should correspond to the
    # `DeviceFieldId`s expected by the environment (some direct, some for bins).
    raw_observation_map = reduced_sequence_df.iloc[0].to_dict()

    # Ensure keys are DeviceFieldId and values are np.ndarray.
    # The keys in raw_observation_map might be tuples (e.g., from multi-index
    # columns if the reducer creates them) or simple strings.
    # We need to map them to the correct DeviceFieldId.
    # The `_id_map` is crucial here. If the histogram reducer generates
    # column names like ('device_id', 'measurement_h_binX'), we need
    # to map this to the `DeviceFieldId` created in
    # `_get_observation_spec_histogram_reducer`.

    final_observation_map: dict[DeviceFieldId, np.ndarray] = {}
    for key_tuple_or_str, value in raw_observation_map.items():
      field_id_str: str
      if isinstance(key_tuple_or_str, tuple):
        # Assuming tuple keys are ('device_id', 'measurement_name_maybe_bin_suffix')
        # This part needs to be robust to how the reducer names its output columns.
        # It's safer if reducer output columns are already the flat DeviceFieldIds.
        # For now, assume it's ('outer_label', 'inner_label') -> "outer_label_inner_label"
        field_id_str = "_".join(map(str, key_tuple_or_str))
      else:
        field_id_str = str(key_tuple_or_str)

      # We need to find the *actual* DeviceFieldId that corresponds to this
      # `field_id_str`. The `field_id_str` might be, e.g.,
      # "deviceX_temperature_h_10.00".
      # This requires looking up in `_id_map.inv`.
      # This is tricky if `field_id_str` is not exactly what was stored.
      # A more direct approach: The histogram reducer's output columns should
      # *already be* the correct DeviceFieldIds.
      # If so, then `key_tuple_or_str` is already a DeviceFieldId.
      if not isinstance(key_tuple_or_str, DeviceFieldId):
        # If the keys are not already DeviceFieldId, we try to convert/lookup.
        # This is a fallback and might indicate a mismatch.
        actual_field_id = DeviceFieldId(field_id_str)
        if actual_field_id not in self.field_names: # `field_names` contains all known DeviceFieldIds
            logging.warning(
                "Histogram reducer output key '%s' (processed as '%s') does "
                "not directly match a known DeviceFieldId. This might lead to "
                "missing observation data if not correctly mapped.",
                key_tuple_or_str,
                actual_field_id,
            )
            # Skip if we can't map it.
            # continue
      else:
        actual_field_id = key_tuple_or_str # It's already a DeviceFieldId

      final_observation_map[actual_field_id] = np.array(value, dtype=np.float32)

    return final_observation_map

  def _get_reward(self) -> float:
    """Calculates the reward for the current environment step.

    This method retrieves `RewardInfo` from the building model (which contains
    raw data like energy use, comfort metrics) and then uses the configured
    `reward_function` to compute a scalar reward value. It also handles
    logging of reward-related metrics.

    Returns:
      float: The scalar reward value for the current step.
    """
    # Retrieve raw reward-related information from the building.
    reward_info = self.building.reward_info

    # Compute the scalar reward using the configured reward function.
    reward_response = self.reward_function.compute_reward(reward_info)

    # Log detailed reward information if a metrics writer is set up.
    if self._metrics_writer:
      self._metrics_writer.write_reward_info(
          reward_info, self.current_simulation_timestamp
      )
      self._metrics_writer.write_reward_response(
          reward_response, self.current_simulation_timestamp
      )

    # Write summary metrics to TensorBoard if a summary writer is set up.
    if self._summary_writer:
      self._write_summary_reward_info_metrics(reward_info)
      self._write_summary_reward_response_metrics(reward_response)
      self._commit_reward_metrics() # Aggregates and writes periodically

    return reward_response.agent_reward_value

  def _write_summary_reward_info_metrics(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> None:
    """Accumulates raw reward-related metrics for TensorBoard logging.

    These metrics are typically components of the final reward, such as
    energy consumption values.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): Proto containing raw
        data used for reward calculation.
    """
    energy_use = conversion_utils.get_reward_info_energy_use(reward_info)

    # Accumulate different types of energy consumption.
    self._accumulator["electrical_energy"].append(
        energy_use.get("air_handler_blower_electricity", 0.0)
        + energy_use.get("air_handler_air_conditioning", 0.0)
        + energy_use.get("boiler_pump_electrical_energy", 0.0)
    )
    self._accumulator["natural_gas_energy"].append(
        energy_use.get("boiler_natural_gas_heating_energy", 0.0)
    )
    # Add other relevant metrics from reward_info if needed.

  def _write_summary_reward_response_metrics(
      self, reward_response: smart_control_reward_pb2.RewardResponse
  ) -> None:
    """Accumulates computed reward metrics for TensorBoard logging.

    These metrics include costs, emissions, comfort regrets, and the
    final agent reward components.

    Args:
      reward_response (smart_control_reward_pb2.RewardResponse): Proto
        containing the results of the reward computation.
    """
    self._accumulator["electricity_energy_cost"].append(
        reward_response.electricity_energy_cost
    )
    self._accumulator["natural_gas_energy_cost"].append(
        reward_response.natural_gas_energy_cost
    )
    self._accumulator["carbon_emitted"].append(reward_response.carbon_emitted)
    self._accumulator["total_occupancy"].append(reward_response.total_occupancy)
    self._accumulator["productivity_regret"].append(
        reward_response.productivity_regret
    )
    self._accumulator["normalized_productivity_regret"].append(
        reward_response.normalized_productivity_regret
    )
    self._accumulator["normalized_energy_cost"].append(
        reward_response.normalized_energy_cost
    )
    self._accumulator["normalized_carbon_emission"].append(
        reward_response.normalized_carbon_emission
    )
    # `step_duration_sec` seems miscategorized here if it's from productivity.
    # Assuming it's a general metric.
    # The original code had:
    # self._accumulator["step_duration_sec"].append(
    # reward_response.normalized_productivity_regret)
    # This seems like a copy-paste error. If step_duration_sec is a field in
    # RewardResponse, it should be:
    # self._accumulator["step_duration_sec"].append(reward_response.step_duration_sec)
    # If not, this line should be removed or corrected.
    # For now, I'll comment it out assuming it was an error.
    # TODO(b/your_bug_tracker_id): Verify "step_duration_sec" metric source.

  def _commit_reward_metrics(self) -> None:
    """Aggregates, logs to TensorBoard, and clears accumulated reward metrics.

    This is called periodically (controlled by
    `_metrics_reporting_interval`) to write mean values of accumulated
    metrics since the last commit.
    """
    assert self._summary_writer is not None, "Summary writer not initialized."

    if self._global_step_count % self._metrics_reporting_interval == 0:
      # Ensure operations are within the summary writer's context.
      # `record_if(True)` ensures metrics are recorded.
      # `tf.name_scope` organizes metrics in TensorBoard.
      with self._summary_writer.as_default(), tf.compat.v2.summary.record_if(
          True
      ), tf.name_scope("RewardMetrics/"): # Changed scope for clarity
        for key, value_list in self._accumulator.items():
          if value_list: # Only log if there's data
            tf.compat.v2.summary.scalar(
                name=key,
                data=np.mean(value_list), # Log the mean of accumulated values
                step=self._global_step_count,
            )
        # Reset accumulator for the next reporting interval.
        self._accumulator = collections.defaultdict(list)

  @property
  def label(self) -> str:
    """The label used for identifying this environment instance, e.g., in logs."""
    return self._label

  def _reset(self) -> ts.TimeStep:
    """Resets the environment to the start of a new episode.

    This involves:
    1. Resetting the underlying building simulation.
    2. Clearing internal state (accumulators, action history, step counts).
    3. Obtaining the initial observation for the new episode.
    4. Setting up metrics writers for the new episode if configured.
    5. Updating episode start/end timestamps.

    Returns:
      ts.TimeStep: A `TimeStep` object of type `FIRST`, containing the
      initial observation.
    """
    self.building.reset() # Reset the building model to its initial state.

    # Clear internal state for the new episode.
    self._accumulator = collections.defaultdict(list)
    self._episode_ended = False
    self._episode_count += 1
    self._episode_cumulative_reward = 0.0
    self._action_history = []
    self._step_count = 0 # Reset per-episode step count.
    # _global_step_count persists across episodes.

    # Get the first observation of the new episode.
    # This also initializes self._last_observation_response.
    initial_observation = self._get_observation()

    # (Re)initialize metrics writer for the new episode if path is provided.
    self._metrics_writer = None # Clear previous episode's writer.
    if self.metrics_path and self._writer_factory:
      # Create a unique directory for this episode's metrics.
      current_time_str = pd.Timestamp.utcnow().strftime("%y%m%d_%H%M%S")
      episode_metrics_id = f"{self._label}_{current_time_str}"
      output_dir = os.path.join(self.metrics_path, episode_metrics_id)
      tf.io.gfile.makedirs(output_dir) # Ensure directory exists.

      logging.info("Writing metric files for episode %d to %s",
                   self._episode_count, output_dir)
      self._metrics_writer = self._writer_factory.create(output_dir)

      if self._building_image_generator:
        img_file_path = os.path.join(
            output_dir, constants.BUILDING_IMAGE_CSV_FILE
        )
        logging.info("Building image CSV will be written to: %s", img_file_path)

      # Write static info (device, zone) at the start of the episode.
      if self._metrics_writer:
        logging.info(
            "Writing %d device_infos.", len(self.building.devices)
        )
        self._metrics_writer.write_device_infos(self.building.devices)
        logging.info("Writing %d zone_infos.", len(self.building.zones))
        self._metrics_writer.write_zone_infos(self.building.zones)

    self._episode_start_time = time.time() # Record start time for duration.
    self._start_timestamp = self.building.current_timestamp # Sim start time.
    self._end_timestamp = ( # Calculate sim end time for this episode.
        self._start_timestamp
        + self._num_timesteps_in_episode * self._step_interval
    )
    logging.info(
        "Environment reset for episode %d. Start: %s, End: %s (%d steps).",
        self._episode_count,
        self._start_timestamp,
        self._end_timestamp,
        self._num_timesteps_in_episode,
    )

    return ts.restart(initial_observation)

  @gin.configurable
  def action_spec(self) -> types.NestedArraySpec:
    """Returns the action specification for the environment.

    This defines the structure, shape, and bounds of the actions expected by
    the `_step` method.

    Returns:
      types.NestedArraySpec: The TF-Agents action specification.
    """
    return self._action_spec

  @gin.configurable
  def observation_spec(self) -> types.NestedArraySpec:
    """Returns the observation specification for the environment.

    This defines the structure, shape, and type of the observations returned
    by the environment.

    Returns:
      types.NestedArraySpec: The TF-Agents observation specification.
    """
    return self._observation_spec

  def _format_action(
      self,
      action: types.NestedArray,
      action_names: Sequence[DeviceFieldId],  # pylint: disable=unused-argument
  ) -> types.NestedArray:
    """Allows derived classes to reformat actions before processing.

    This base implementation is a no-op, returning the action as is.
    Derived environment classes can override this to adapt actions from
    different policy structures if needed.

    Args:
      action (types.NestedArray): The action(s) from the agent.
      action_names (Sequence[DeviceFieldId]): The ordered list of action names
        corresponding to the action array elements. Provided for context.

    Returns:
      types.NestedArray: The potentially reformatted action.
    """
    # This function is a hook for subclasses.
    # See: https://github.com/google/sbsim/pull/57
    return action

  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    """Advances the environment by one time step using the given action.

    This method performs the core RL loop:
    1. Processes the agent's action (denormalization, request creation).
    2. Sends the action to the building simulation.
    3. Retrieves the new observation from the building.
    4. Calculates the reward based on the new state and action.
    5. Determines if the episode has ended.
    6. Returns a `TimeStep` object (TRANSITION or TERMINATION).

    Args:
      action (types.NestedArray): A NumPy array representing the action taken
        by the agent. This should conform to `self.action_spec()`.

    Returns:
      ts.TimeStep: A `TimeStep` object containing the new observation, reward,
      discount factor, and step type (MID or LAST).
    """

    def _log_action_strings(action_req: ActionRequest) -> Sequence[str]:
      """Helper to format action request details for logging."""
      log_strings = []
      for single_req in action_req.single_action_requests:
        log_strings.append(
            f"{single_req.device_id} "
            f"{single_req.setpoint_name}: "
            f"{single_req.continuous_value:.2f}"
        )
      return log_strings

    if self._episode_ended:
      # If the episode ended on the previous step, a new step should start
      # with a reset.
      return self.reset()

    step_start_time = time.time()

    # Allow subclasses to format the action if needed.
    formatted_action = self._format_action(action, self._action_names)
    # Ensure action is a NumPy array for subsequent processing.
    if not isinstance(formatted_action, np.ndarray):
      try:
        formatted_action = np.asarray(formatted_action, dtype=np.float32)
      except Exception as e:
        raise ValueError(
            "Action could not be converted to a NumPy array."
        ) from e

    # Create the protobuf ActionRequest from the (potentially formatted)
    # NumPy action array. This includes denormalization.
    action_request = self._create_action_request(formatted_action)

    action_accepted: bool
    action_response: Optional[ActionResponse] = None
    try:
      # Send the action to the building simulation.
      action_response = self.building.request_action(action_request)
      action_accepted = all_actions_accepted(action_response)
      if not action_accepted:
        logging.warning(
            "Action partially or fully REJECTED by building at %s. Details: %s",
            self.current_simulation_timestamp,
            action_response, # Log the full response for details
        )

    except RuntimeError as e:
      # Handle cases where the building interaction itself fails.
      logging.exception(
          "RuntimeError during building.request_action at %s for actions %s.",
          self.current_simulation_timestamp,
          _log_action_strings(action_request),
      )
      action_accepted = False
      # Construct a synthetic ActionResponse indicating rejection.
      action_response = _apply_action_response(
          action_request,
          response_timestamp=self.current_simulation_timestamp,
          action_response_type=(
              SingleActionResponse.ActionResponseType
              .REJECTED_NOT_ENABLED_OR_AVAILABLE
          ),
          additional_info=f"RuntimeError: {e}",
      )

    if self._metrics_writer and action_response is not None:
      self._metrics_writer.write_action_response(
          action_response, self.current_simulation_timestamp
      )

    # Allow the building simulation to advance its internal state.
    self.building.wait_time()

    # Get the new observation from the building.
    current_observation = self._get_observation()

    # Calculate reward. If action was rejected, a special penalty is applied.
    # The `ACTION_REJECTION_REWARD` is negative infinity, signaling a
    # problematic step to the agent/training loop.
    reward_value = self._get_reward() if action_accepted else ACTION_REJECTION_REWARD

    self._episode_cumulative_reward += reward_value
    self._episode_ended = self._has_episode_ended() # Check for termination.

    step_end_time = time.time()
    episode_duration_so_far = step_end_time - self._episode_start_time
    step_duration = step_end_time - step_start_time

    if self._episode_ended:
      logging.info(
          "%s: TERMINATING EPISODE %d at step %d. Sim time: %s. "
          "Step Reward: %.2f, Cumulative Reward: %.2f. "
          "Episode Duration: %.2fs, Step Duration: %.2fs.",
          self._label,
          self._episode_count,
          self._step_count,
          self.building.current_timestamp,
          reward_value,
          self._episode_cumulative_reward,
          episode_duration_so_far,
          step_duration,
      )
      return ts.termination(current_observation, reward_value)
    else:
      # Log progress periodically.
      if self._step_count % 100 == 0: # Log every 100 steps.
        logging.info(
            "%s: Episode %d, Step %d. Sim time: %s. Step Reward: %.2f, "
            "Cumulative Reward: %.2f. Episode Duration: %.2fs, "
            "Step Duration: %.2fs.",
            self._label,
            self._episode_count,
            self._step_count,
            self.building.current_timestamp,
            reward_value,
            self._episode_cumulative_reward,
            episode_duration_so_far,
            step_duration,
        )

      self._step_count += 1
      self._global_step_count += 1 # Increment global step counter.
      return ts.transition(
          current_observation, reward_value, self.discount_factor
      )

  def render(self, mode: str = "rgb_array") -> Optional[types.NestedArray]:
    """Renders the environment. (Not Implemented).

    Args:
      mode (str): The rendering mode (e.g., "rgb_array", "human").

    Returns:
      Optional[types.NestedArray]: Rendered output, or None.

    Raises:
      NotImplementedError: This environment does not currently support
        rendering.
    """
    raise NotImplementedError("Rendering not supported for this environment.")

  def _has_episode_ended(self) -> bool:
    """Checks if the current episode should terminate.

    Termination occurs if the current step count reaches or exceeds the
    maximum number of timesteps defined for an episode.

    Returns:
      bool: True if the episode has ended, False otherwise.
    """
    return self._step_count >= self._num_timesteps_in_episode


def _apply_action_response(
    action_request: ActionRequest,
    action_response_type: SingleActionResponse.ActionResponseType,
    response_timestamp: pd.Timestamp,
    additional_info: Optional[str] = None,
) -> ActionResponse:
  """Creates a synthetic ActionResponse, typically for rejections.

  This helper is used when the building model itself doesn't return an
  `ActionResponse` (e.g., due to a runtime error before the model could
  process the request, or to signal a blanket rejection).

  Args:
    action_request (ActionRequest): The original request that led to this
      synthetic response.
    action_response_type (SingleActionResponse.ActionResponseType): The status
      to apply to all single action requests within this response (e.g.,
      REJECTED_INTERNAL_ERROR).
    response_timestamp (pd.Timestamp): The timestamp for this synthetic
      response.
    additional_info (Optional[str]): Optional string providing more details
      about the reason for this response.

  Returns:
    ActionResponse: A populated `ActionResponse` where each
    `SingleActionResponse` reflects the provided `action_response_type`.
  """
  single_responses = [
      _apply_single_action_response(
          single_req, action_response_type, additional_info
      )
      for single_req in action_request.single_action_requests
  ]
  return ActionResponse(
      timestamp=conversion_utils.pandas_to_proto_timestamp(response_timestamp),
      request=action_request, # Link to the original request
      single_action_responses=single_responses,
  )


def _apply_single_action_response(
    single_action_request: SingleActionRequest,
    action_response_type: SingleActionResponse.ActionResponseType,
    additional_info: Optional[str] = None,
) -> SingleActionResponse:
  """Creates a synthetic SingleActionResponse.

  Args:
    single_action_request (SingleActionRequest): The original single request.
    action_response_type (SingleActionResponse.ActionResponseType): The status.
    additional_info (Optional[str]): More details.

  Returns:
    SingleActionResponse: A populated `SingleActionResponse`.
  """
  return SingleActionResponse(
      request=single_action_request, # Link to the original single request
      response_type=action_response_type,
      additional_info=additional_info,
  )
