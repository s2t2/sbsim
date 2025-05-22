"""Controllable building RL environment to interact with TF-Agents.

RL environment where the agent is able to control various
setpoints with the goal of making the HVAC system more efficient.

Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

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
  """Checks if all actions in an ActionResponse were accepted.

  Iterates through the `single_action_responses` in an `ActionResponse`
  and returns True if all of them have a `response_type` of `ACCEPTED`.

  Args:
    action_response: An `ActionResponse` proto message.

  Returns:
    True if all single action requests were accepted, False otherwise.
  """
  return all(
      single_action_response.response_type == SingleActionResponse.ACCEPTED
      for single_action_response in action_response.single_action_responses
  )


def replace_missing_observations_past(
    current_observation_response: ObservationResponse,
    past_observation_response: Optional[ObservationResponse],
) -> ObservationResponse:
  """Fills missing observations in the current response with past values.

  In smart building environments, sensor readings or other observations might
  occasionally be missing. This function ensures that the agent always receives
  a complete set of observations by imputing missing values from the
  `past_observation_response`.

  The function identifies missing observations by comparing the requested
  observations (from `current_observation_response.request`) with the valid
  observations actually provided in `current_observation_response`.

  Args:
    current_observation_response: The `ObservationResponse` from the current
      timestep, which may contain missing or invalid observations.
    past_observation_response: An optional `ObservationResponse` from a
      previous timestep. If provided, valid observations from this response
      will be used to fill in any gaps in `current_observation_response`.

  Returns:
    An `ObservationResponse` where missing or invalid observations in the
    `current_observation_response` have been replaced with corresponding
    valid observations from `past_observation_response`. If no past
    observation is available to fill a missing value, the original (invalid)
    observation is retained, and a warning is logged.

  Raises:
    ValueError: If `current_observation_response` contains missing observations
      and `past_observation_response` is `None`, indicating no data is
      available for imputation.
  """

  def get_observation_request_tuples(
      observation_request: ObservationRequest,
  ) -> set[DeviceMeasurementTuple]:
    return set([
        (request.device_id, request.measurement_name)
        for request in observation_request.single_observation_requests
    ])

  def get_observation_response_mapping(
      observation_response: ObservationResponse,
  ) -> dict[
      DeviceMeasurementTuple,
      SingleObservationResponse,
  ]:
    """Converts an ObservationResponse into a dict of single observations."""
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
    """Checks that the past observation is available, and raises ValueError."""
    if not past_observation_response:
      # If there is not a past response, then provide a detailed log entry and
      # raise a ValueError.
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
    """Returns device/measurements set for requests that weren't provided."""

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
    """Checks a single observation response and fills in when invalid."""
    if single_observation_response.observation_valid:
      updated_single_observation_response = single_observation_response
    # If it's not valid, then use the past observation to fill in the gap.
    else:
      request = single_observation_response.single_observation_request
      missing_observation = (request.device_id, request.measurement_name)
      updated_single_observation_response = past_observation_response_mapping[
          missing_observation
      ]

      logging.warning(
          "Missing or invalid observation response for %s %s; replacing it with"
          " past observation.",
          missing_observation[0],
          missing_observation[1],
      )
    return updated_single_observation_response

  # Compare what's in the request to what was returned in the response.
  # Put any missing or invalid responses into the missing observations list.
  missing_observations = get_missing_observations(current_observation_response)

  if missing_observations:
    # If there are missing observations and we have a past ObservationRespose,
    # filling the missing values from the past response.
    # If there are no missing observations, just return the original
    # ObservationResponse.
    check_valid_past_observation(
        past_observation_response, missing_observations
    )

    updated_single_observation_responses = []

    past_observation_response_mapping = get_observation_response_mapping(
        past_observation_response
    )

    # Maintain the same ordering between the requests and responses.
    for (
        single_observation_response
    ) in current_observation_response.single_observation_responses:
      # If the observation is valid, just add it to the updated list.
      updated_single_observation_response = update_single_observation_response(
          single_observation_response, past_observation_response_mapping
      )

      updated_single_observation_responses.append(
          updated_single_observation_response
      )
      # Create a new observation response that combines both the valid current
      # observations and the valid past observations when the current is
      # invalid.
      current_observation_response = copy.deepcopy(current_observation_response)
      del current_observation_response.single_observation_responses[:]
      current_observation_response.single_observation_responses.extend(
          updated_single_observation_responses
      )

  return current_observation_response


def compute_action_regularization_cost(
    action_history: Sequence[np.ndarray],
) -> float:
  """Calculates a cost for action smoothness based on recent action history.

  This cost penalizes large changes in actions between consecutive timesteps.
  It is calculated as the L2 norm (Euclidean distance) of the difference
  between the last two actions in the `action_history`.

  Args:
    action_history: A sequence of NumPy arrays, where each array represents
      the action taken at a timestep. The sequence is ordered chronologically.

  Returns:
    The L2 norm of the difference between the last two actions if at least
    two actions are present in the history. Returns 0.0 if there is only one
    action (no change to measure) or if the history is empty.

  Raises:
    ValueError: If the shapes of the last two actions in `action_history`
      do not match.
  """
  if len(action_history) > 1:
    if action_history[-2].shape != action_history[-1].shape:
      raise ValueError("Action history shapes do not match.")
    return np.linalg.norm(
        action_history[-2] - action_history[-1], axis=0, ord=2
    )
  else:
    return 0.0


@gin.configurable
class ActionConfig:
  """Configuration for action normalization in the environment.

  This class defines how actions, specifically setpoints for various devices,
  are normalized. It maps `DeviceFieldId` (a unique identifier for a device's
  setpoint) to a corresponding `BaseActionNormalizer` instance.

  The environment will only allow control over setpoints that are explicitly
  configured in this `ActionConfig`.

  Attributes:
    action_normalizers: A mapping from `DeviceFieldId` to an instance of
      `base_normalizer.BaseActionNormalizer`. This normalizer defines how the
      raw action values from the agent are converted to physical setpoint values
      for the building simulation, and vice-versa.

  Example:
    ```python
    from smart_control.models import base_normalizer

    action_config = ActionConfig(
        action_normalizers={
            DeviceFieldId('boiler_0_supply_water_setpoint'):
                base_normalizer.ContinuousBaseActionNormalizer(
                    min_val=60.0, max_val=80.0
                ),
            DeviceFieldId('ahu_1_damper_position'):
                base_normalizer.DiscreteBaseActionNormalizer(
                    possible_values=[0.0, 0.5, 1.0]
                ),
        }
    )
    ```
  This example configures normalization for two setpoints:
  - The 'supply_water_setpoint' of 'boiler_0' is continuous between 60.0 and 80.0.
  - The 'damper_position' of 'ahu_1' can take discrete values 0.0, 0.5, or 1.0.
  """

  def __init__(self, action_normalizers: ActionNormalizerMap):
    """Initializes ActionConfig.

    Args:
      action_normalizers: A dictionary mapping `DeviceFieldId` strings to
        `BaseActionNormalizer` instances.
    """
    self.action_normalizers = action_normalizers

  def get_action_normalizer(
      self, setpoint_name: FieldName
  ) -> Optional[base_normalizer.BaseActionNormalizer]:
    """Retrieves the action normalizer for a given setpoint name.

    Note: This method currently uses `FieldName` which might represent
    a generic field name. It's typically expected that a more specific
    `DeviceFieldId` (which includes the device ID) would be used to look up
    normalizers, as normalizers are usually specific to a device-setpoint pair.
    The current implementation might lead to ambiguity if multiple devices
    share the same `setpoint_name` but require different normalization.

    Args:
      setpoint_name: The name of the setpoint (e.g., 'supply_water_setpoint').
        Ideally, this should be a `DeviceFieldId`.

    Returns:
      The `BaseActionNormalizer` instance associated with the given
      `setpoint_name`, or `None` if no normalizer is found for that name.
    """
    return self.action_normalizers.get(DeviceFieldId(setpoint_name))


def generate_field_id(
    device: DeviceId, field: FieldName, id_map: bidict.bidict
) -> DeviceFieldId:
  """Generates a unique string identifier for a device field.

  This function creates a `DeviceFieldId` by combining a `device` identifier
  (e.g., 'boiler_0') and a `field` name (e.g., 'supply_water_setpoint').
  The format is typically `device_field`.

  It ensures uniqueness by checking against an existing `id_map` (a
  bidirectional dictionary mapping `(DeviceId, FieldName)` tuples to
  `DeviceFieldId` strings).

  - If the exact `(device, field)` pair already exists in `id_map`, its
    existing `DeviceFieldId` is returned. This can happen if an observable
    and an actionable field share the same name for the same device.
  - If the generated `device_field` string collides with an existing
    `DeviceFieldId` that was generated from a *different* `(device, field)`
    pair, an integer suffix (e.g., `_1`, `_2`) is appended to the new ID
    to ensure uniqueness.

  Args:
    device: The identifier of the device (e.g., 'ahu_1', 'sensor_temp_room_a').
    field: The name of the measurement or setpoint associated with the device
      (e.g., 'zone_temperature', 'damper_command').
    id_map: A `bidict.bidict` instance that stores existing mappings between
      `(device, field)` tuples and their unique `DeviceFieldId` strings. This
      map is used to check for pre-existing IDs and to resolve collisions.

  Returns:
    A unique `DeviceFieldId` string for the given device and field.

  Examples:
    >>> id_map = bidict.bidict()
    >>> generate_field_id(DeviceId('a_b'), FieldName('c'), id_map)
    'a_b_c'
    >>> id_map[('a_b', 'c')] = 'a_b_c' # Simulate adding to map
    >>> generate_field_id(DeviceId('a_b'), FieldName('c'), id_map) # Existing pair
    'a_b_c'
    >>> generate_field_id(DeviceId('a'), FieldName('b_c'), id_map) # Collision
    'a_b_c_1'
  """
  if (device, field) in id_map:
    # This case handles when the same (device, field) pair is requested again,
    # or if an observable and an actionable field for the same device have the
    # same name.
    return id_map[(device, field)]

  new_id = f"{device}_{field}"
  counter = 0

  # Check for duplicates.
  while new_id in id_map.inv:
    counter += 1
    new_id = f"{device}_{field}_{counter}"

  return DeviceFieldId(new_id)


@gin.configurable
class Environment(py_environment.PyEnvironment):
  """A reinforcement learning environment for smart building control.

  This environment interfaces with a building simulation (`base_building.BaseBuilding`)
  and allows a TF-Agents agent to learn control strategies for optimizing
  building operations (e.g., energy consumption, occupant comfort).

  Key features:
  - **State Representation:** Observations include sensor readings from the
    building (e.g., temperatures, CO2 levels), time-based features (hour of
    day, day of week), and occupancy information. Observations can be
    normalized and optionally reduced using histogram-based techniques.
  - **Action Space:** Actions typically involve adjusting setpoints for HVAC
    components (e.g., supply air temperature, damper positions). Actions are
    normalized based on configurations provided in `ActionConfig`.
  - **Reward Calculation:** A customizable reward function (`base_reward_function.BaseRewardFunction`)
    evaluates the agent's performance based on metrics like energy usage,
    comfort levels, and operational costs.
  - **Episode Management:** The environment handles episode initialization,
    stepping through time, and termination based on a defined episode duration.
  - **Metrics and Logging:** Supports extensive logging of environment data,
    including observations, actions, rewards, and building state, to facilitate
    analysis and debugging. It can also generate building image encodings.

  The environment is configurable using Gin, allowing for flexible setup of
  its components (building model, reward function, normalizers, etc.).
  """

  def __init__(
      self,
      building: base_building.BaseBuilding,
      reward_function: base_reward_function.BaseRewardFunction,
      observation_normalizer: base_normalizer.BaseObservationNormalizer,
      action_config: ActionConfig,
      discount_factor: float = 1,
      metrics_path: str | None = None,
      num_days_in_episode: int = 3,
      device_action_tuples: Sequence[DeviceActionTuple] | None = None,
      default_actions: DefaultActions | None = None,
      metrics_reporting_interval: float = 100,
      label: str = "episode_metrics",
      num_hod_features: int = 1,
      num_dow_features: int = 1,
      occupancy_normalization_constant: float = 0.0,
      run_command_predictors: (
          Sequence[run_command_predictor.BaseRunCommandPredictor] | None
      ) = None,
      observation_histogram_reducer: (
          histogram_reducer.HistogramReducer | None
      ) = None,
      time_zone: str = "US/Pacific",
      image_generator: (
          building_image_generator.BuildingImageGenerator | None
      ) = None,
      step_interval: pd.Timedelta = pd.Timedelta(5, unit="minutes"),
      writer_factory: writer_lib.BaseWriterFactory | None = None,
  ) -> None:
    """Initializes the smart building control environment.

    Args:
      building: An instance of `base_building.BaseBuilding` representing the
        building to be controlled. This object simulates the building's
        thermodynamics and systems.
      reward_function: An instance of
        `base_reward_function.BaseRewardFunction` that defines how the agent's
        reward is calculated at each step.
      observation_normalizer: An instance of
        `base_normalizer.BaseObservationNormalizer` used to normalize the raw
        observations from the building before they are passed to the agent.
      action_config: An `ActionConfig` instance that defines the controllable
        setpoints, their normalization, and their bounds.
      discount_factor: The discount factor (gamma) for future rewards, typically
        between 0 and 1. Defaults to 1.
      metrics_path: An optional string specifying the directory path where
        environment metrics (like observations, actions, rewards) will be
        written. If `None`, metrics are not written to files.
      num_days_in_episode: The duration of each RL episode in days.
        Defaults to 3.
      device_action_tuples: An optional sequence of `DeviceActionTuple`
        (device_id, setpoint_name) that explicitly defines which setpoints are
        controllable by the agent. If `None`, the environment attempts to
        derive controllable setpoints from the `building.devices` and
        `action_config`.
      default_actions: An optional `DefaultActions` mapping
        (`DeviceFieldId` to float) specifying the initial actions to take at the
        beginning of an episode or when actions are reset.
      metrics_reporting_interval: The interval (in environment steps) at which
        aggregated metrics are reported to TensorBoard. Defaults to 100.
      label: A string label used as a prefix for episode output directories if
        `metrics_path` is specified. Defaults to "episode_metrics".
      num_hod_features: The number of sine/cosine pairs used to encode the
        hour of the day as a cyclical feature. Defaults to 1.
      num_dow_features: The number of sine/cosine pairs used to encode the
        day of the week as a cyclical feature. Defaults to 1.
      occupancy_normalization_constant: A constant used in the normalization
        of the occupancy signal. Defaults to 0.0.
      run_command_predictors: An optional sequence of
        `run_command_predictor.BaseRunCommandPredictor` instances. These can be
        used to predict optimal on/off commands for equipment based on the
        agent's actions or other environmental factors.
      observation_histogram_reducer: An optional
        `histogram_reducer.HistogramReducer` instance used to reduce the
        dimensionality of the observation space by binning certain features
        into histograms.
      time_zone: The time zone of the building and environment (e.g.,
        "US/Pacific"). Defaults to "US/Pacific".
      image_generator: An optional
        `building_image_generator.BuildingImageGenerator` used to create image
        encodings (e.g., heatmaps) from the building's observation responses.
      step_interval: A `pd.Timedelta` object specifying the duration between
        consecutive steps in the environment. Defaults to 5 minutes.
      writer_factory: An optional `writer_lib.BaseWriterFactory` used to create
        metric writers if `metrics_path` is provided.

    Raises:
      ValueError: If `discount_factor` is not in the range (0, 1].
    """
    super().__init__()

    self.building: base_building.BaseBuilding = building
    self._time_zone = time_zone
    self._device_action_tuples: Optional[Sequence[DeviceActionTuple]] = (
        device_action_tuples
    )
    self.reward_function: base_reward_function.BaseRewardFunction = (
        reward_function
    )
    self._observation_histogram_reducer = observation_histogram_reducer
    self.discount_factor: float = discount_factor
    self._step_count: int = 0
    self._global_step_count: int = 0
    self._episode_count: int = 0
    self._episode_cumulative_reward: float = 0
    self._last_log_timestamp: float = 0.0
    self._observation_normalizer: base_normalizer.BaseObservationNormalizer = (
        observation_normalizer
    )
    self._start_timestamp: pd.Timestamp = self.building.current_timestamp
    self._action_history = []
    self._end_timestamp: pd.Timestamp = self._start_timestamp + pd.Timedelta(
        num_days_in_episode, unit="days"
    )
    self._step_interval = step_interval
    self._num_timesteps_in_episode = int(
        (self._end_timestamp - self._start_timestamp) / self._step_interval
    )
    self._metrics = plot_utils.init_metrics()
    logging.info(
        "Episode starts at %s and ends at %s; % d timesteps.",
        self._start_timestamp,
        self._end_timestamp,
        self._num_timesteps_in_episode,
    )

    self._id_map = bidict.bidict()

    if self.discount_factor <= 0 or self.discount_factor > 1:
      raise ValueError("Discount factor must be in (0,1]")

    self.metrics_path: Optional[str] = metrics_path
    self._writer_factory: Optional[writer_lib.BaseWriterFactory] = (
        writer_factory
    )
    self._metrics_writer: Optional[writer_lib.BaseWriter] = None
    self._summary_writer = None
    self._label = label
    self._num_dow_features = num_dow_features
    self._num_hod_features = num_hod_features
    # Retain the last observation to fill in missing or invalid values.
    self._last_observation_response: Optional[ObservationResponse] = None

    if self.discount_factor <= 0 or self.discount_factor > 1:
      raise ValueError("Discount factor must be in (0,1]")

    if device_action_tuples is not None:
      self._action_spec, self.action_normalizers, self._action_names = (
          self._get_action_spec_and_normalizers_from_device_action_tuples(
              action_config=action_config,
              device_action_tuples=device_action_tuples,
          )
      )
    else:
      self._action_spec, self.action_normalizers, self._action_names = (
          self._get_action_spec_and_normalizers(action_config, building.devices)
      )

    logging.info("Action Names %s", self._action_names)

    self._auxiliary_features = self._get_auxiliary_features_labels(
        self._num_hod_features, self._num_dow_features
    )
    logging.info("Auxiliary Features %s", self._auxiliary_features)

    self._observation_spec, self.field_names = self._get_observation_spec(
        building.devices
    )
    logging.info("Observation Spec %s", self._observation_spec)

    logging.info("%s FIELD NAMES (%d)", self._label, len(self.field_names))
    for i, fn in enumerate(self.field_names):
      logging.info("Field %d: %s", i, fn)

    self._episode_ended = False
    self._episode_start_time = time.time()

    self._default_policy_values = (
        self._normalize_default_actions(default_actions)
        if default_actions
        else tf.constant([])
    )

    self._accumulator = collections.defaultdict(list)
    self._metrics_reporting_interval = metrics_reporting_interval
    # Since the request will not change (i.e., feature vector is fixed),
    # just define a single ObservationRequest as a template for all requests.
    self._observation_request = self._get_observation_request(building.devices)
    self.occupancy_normalization_constant = occupancy_normalization_constant
    if run_command_predictors is None:
      self._run_command_predictors = None
    else:
      self._run_command_predictors = list(run_command_predictors)

    self._building_image_generator = image_generator

  def set_summary_writer(self, summary_path: str) -> None:
    """Sets up a TensorFlow summary writer for logging metrics to TensorBoard.

    Args:
      summary_path: The directory path where TensorBoard logs will be written.
    """
    self._summary_writer = tf.compat.v2.summary.create_file_writer(
        summary_path, flush_millis=10000
    )

  @property
  def steps_per_episode(self) -> int:
    """The total number of simulation steps in a single episode."""
    return (
        self._end_timestamp - self._start_timestamp
    ).total_seconds() // self.building.time_step_sec

  @property
  def start_timestamp(self) -> pd.Timestamp:
    """The timestamp marking the beginning of the current episode."""
    return self._start_timestamp

  @property
  def end_timestamp(self) -> pd.Timestamp:
    """The timestamp marking the end of the current episode."""
    return self._end_timestamp

  @end_timestamp.setter
  def end_timestamp(self, value: pd.Timestamp):
    """Sets the timestamp that marks the end of the current episode.

    Args:
      value: A `pd.Timestamp` to set as the end timestamp.
    """
    self._end_timestamp = value

  @property
  def default_policy_values(self) -> tf.Tensor:
    """Normalized default action values used at the start of an episode.

    Returns:
      A TensorFlow constant tensor containing the normalized default actions.
      Returns an empty tensor if no default actions were specified.
    """
    return self._default_policy_values

  def _get_observation_request(
      self, devices: Sequence[DeviceInfo]
  ) -> ObservationRequest:
    """Creates a template `ObservationRequest` based on device information.

    This request asks for all `observable_fields` from all specified `devices`.
    The timestamp is not set here and should be set before each use.

    Args:
      devices: A sequence of `DeviceInfo` protocol buffers, each describing
        a device and its observable fields.

    Returns:
      An `ObservationRequest` proto message configured to request all
      observable fields from the given devices.
    """
    observation_request = ObservationRequest()
    for device in sorted(devices, key=lambda x: x.device_id):
      for measurement_name in sorted(device.observable_fields):
        device_id = device.device_id
        observation_request.single_observation_requests.add(
            device_id=device_id, measurement_name=measurement_name
        )
    return observation_request

  def _get_auxiliary_features_labels(
      self, num_hod_features: int, num_dow_features: int
  ) -> Sequence[str]:
    """Generates labels for auxiliary time-based and occupancy features.

    These labels correspond to features added to the observation space, such as
    sine/cosine transformations of the hour of day (HOD) and day of week (DOW),
    comfort mode indicators, and number of occupants.

    Args:
      num_hod_features: Number of sine/cosine pairs for hour-of-day features.
      num_dow_features: Number of sine/cosine pairs for day-of-week features.

    Returns:
      A sequence of strings, where each string is a label for an auxiliary
      feature (e.g., 'sin_hod_0', 'cos_dow_1', 'comfort_mode_now').
    """
    return (
        [
            f"{tup[0]}_{tup[1]}"
            for tup in regression_building_utils.get_time_feature_names(
                num_hod_features, HOD_LABEL
            )
        ]
        + [
            f"{tup[0]}_{tup[1]}"
            for tup in regression_building_utils.get_time_feature_names(
                num_dow_features, DOW_LABEL
            )
        ]
        + [COMFORT_MODE_NOW, COMFORT_MODE_SOON, NUM_OCCUPANTS]
    )

  def _normalize_default_actions(self, default_actions: DefaultActions):
    """Converts the default actions into a normalized action array."""

    fixed_actions = []
    for field_id in self._action_names:
      # assert action_name in default_actions

      _, setpoint_name = self._id_map.inv[field_id]
      native_setpoint_value = default_actions[setpoint_name]
      normalized_agent_value = self.action_normalizers[field_id].agent_value(
          native_setpoint_value
      )
      fixed_actions.append(normalized_agent_value)

    """Converts a dictionary of default actions to a normalized tensor.

    The native default action values are transformed into the agent's
    normalized action space (typically [-1, 1]) using the corresponding
    action normalizers.

    Args:
      default_actions: A `DefaultActions` mapping where keys are `FieldName`
        (representing setpoint names) and values are their native default
        values. Note: The type hint `DefaultActions` uses `DeviceFieldId` as
        keys, which might be more accurate if defaults are per device-setpoint.
        The implementation iterates `self._action_names` (which are
        `DeviceFieldId`s) and looks up `setpoint_name` from `self._id_map.inv`.
        This implies `default_actions` should ideally be keyed by `FieldName`
        that can be found in `self._id_map.inv`.

    Returns:
      A TensorFlow constant tensor of float32 values representing the
      normalized default actions, in the order of `self._action_names`.
    """
    fixed_actions = []
    for field_id in self._action_names:
      # Ensure field_id is in the inverse map to get device_id, setpoint_name
      if field_id not in self._id_map.inv:
        raise ValueError(
            f"Field ID {field_id} not found in id_map. Ensure default_actions"
            " are correctly specified and id_map is populated."
        )

      _device_id, setpoint_name = self._id_map.inv[field_id]

      # Ensure the setpoint_name from the id_map is present in default_actions
      if setpoint_name not in default_actions:
        raise ValueError(
            f"Setpoint {setpoint_name} (from field_id {field_id}) not found in"
            " default_actions. Available default_actions keys:"
            f" {list(default_actions.keys())}"
        )

      native_setpoint_value = default_actions[setpoint_name]

      # Ensure the field_id has a corresponding normalizer
      if field_id not in self.action_normalizers:
        raise ValueError(
            f"Action normalizer for field_id {field_id} not found."
        )

      normalized_agent_value = self.action_normalizers[field_id].agent_value(
          native_setpoint_value
      )
      fixed_actions.append(normalized_agent_value)

    return tf.constant(fixed_actions, dtype=tf.float32)

  def _get_action_spec_and_normalizers(
      self,
      action_config: ActionConfig,
      devices: Sequence[DeviceInfo],
  ) -> Tuple[types.ArraySpec, ActionNormalizerMap, Sequence[str]]:
    """Builds the action specification and normalizer map from device info.

    This method defines the agent's action space by inspecting the
    `action_fields` of the provided `devices` and consulting the
    `action_config` for normalization details. Only setpoints included in
    `action_config` become part of the action space.

    Args:
      action_config: An `ActionConfig` instance that provides normalizers
        and specifications for controllable setpoints.
      devices: A sequence of `DeviceInfo` protocol buffers, describing the
        available devices and their action fields (setpoints).

    Returns:
      A tuple containing:
        - action_spec: A `tf_agents.specs.array_spec.BoundedArraySpec` defining
          the shape, dtype (float32), and bounds (-1.0 to 1.0) of the
          normalized action space.
        - action_normalizers: An `ActionNormalizerMap` (dictionary) mapping
          `DeviceFieldId` strings to their corresponding
          `base_normalizer.BaseActionNormalizer` instances.
        - action_names: A sequence of `DeviceFieldId` strings indicating the
          order of actions in the `action_spec` array.

    Raises:
      ValueError: If a device action field has an undefined value type.
      NotImplementedError: If a device action field value type is not
        `VALUE_CONTINUOUS`.
    """

    def _check_value_type_continuous(value: ValueType) -> None:
      """Checks if the value type is continuous, raising error if not."""
      if value == ValueType.VALUE_TYPE_UNDEFINED:
        raise ValueError("Value Type Undefined")
      elif value != ValueType.VALUE_CONTINUOUS:
        raise NotImplementedError("Value Type not supported")

    action_spec = {}
    action_normalizers = {}
    action_names = []
    logging.info(
        "Loading device-setpoint pairs from %d device_infos.", len(devices)
    )
    for device in devices:
      # We need to apply an arbitrary, but consistent ordering the actions
      # within a device. Since device.action_fields is a map and has a random
      # order, we choose to sort the actions within a device alphabetically.
      for setpoint_name in sorted(device.action_fields.keys()):
        value = device.action_fields[setpoint_name]

        device_id = DeviceId(device.device_id)
        setpoint_name = FieldName(setpoint_name)

        # Get BaseActionNormalizer based on device and setpoint_name
        action_normalizer = action_config.get_action_normalizer(setpoint_name)

        # Do not add to action_spec without an action_normalizer.
        if not action_normalizer:
          continue

        field_id = generate_field_id(device_id, setpoint_name, self._id_map)
        self._id_map[(device.device_id, setpoint_name)] = field_id
        action_names.append(field_id)

        _check_value_type_continuous(value)
        field_array_spec = action_normalizer.get_array_spec(field_id)

        action_spec[field_id] = field_array_spec
        action_normalizers[field_id] = action_normalizer

    action_spec = array_spec.BoundedArraySpec(
        shape=(len(action_names),),
        dtype=np.float32,
        name="action",
        minimum=-1.0,
        maximum=1.0,
    )
    logging.info(
        "The action_spec contains %d actions: %s.",
        len(action_names),
        ", ".join(action_names),
    )

    return action_spec, action_normalizers, action_names

  def _get_action_spec_and_normalizers_from_device_action_tuples(
      self,
      action_config: ActionConfig,
      device_action_tuples: Sequence[DeviceActionTuple],
  ) -> Tuple[types.ArraySpec, ActionNormalizerMap, Sequence[str]]:
    """Builds action spec and normalizers from explicit device-action tuples.

    Similar to `_get_action_spec_and_normalizers`, but uses a predefined list
    of `device_action_tuples` instead of inferring from all device fields.
    This allows for more precise control over which setpoints are included in
    the action space.

    Args:
      action_config: An `ActionConfig` instance providing normalizers.
      device_action_tuples: A sequence of `(DeviceCode, Setpoint)` tuples
        explicitly listing the actions to be included in the action space.

    Returns:
      A tuple containing:
        - action_spec: The `BoundedArraySpec` for the actions.
        - action_normalizers: The map from `DeviceFieldId` to normalizers.
        - action_names: The ordered list of `DeviceFieldId`s for the spec.

    Raises:
      ValueError: If an action normalizer is missing in `action_config` for
        any of the specified `device_action_tuples`.
    """
    action_spec = {}
    action_normalizers = {}
    action_names = []
    logging.info(
        "Loading device-setpoint pairs from %d device_action_tuples.",
        len(device_action_tuples),
    )
    for device_action_tuple in device_action_tuples:
      device_id = DeviceId(device_action_tuple[0])
      setpoint_name = FieldName(device_action_tuple[1])

      # Get BaseActionNormalizer based on device and setpoint_name
      action_normalizer = action_config.get_action_normalizer(setpoint_name)

      # Do not add to action_spec without an action_normalizer.
      # TODO(sipple) Include a unit test.
      if not action_normalizer:
        raise ValueError("Missing a normalizer")

      field_id = generate_field_id(device_id, setpoint_name, self._id_map)
      self._id_map[(device_id, setpoint_name)] = field_id
      action_names.append(field_id)

      field_array_spec = action_normalizer.get_array_spec(field_id)
      action_spec[field_id] = field_array_spec
      action_normalizers[field_id] = action_normalizer

    action_spec = array_spec.BoundedArraySpec(
        shape=(len(action_names),),
        dtype=np.float32,
        name="action",
        minimum=-1.0,
        maximum=1.0,
    )
    logging.info(
        "The action_spec from device_action_tuples contains %d actions: %s.",
        len(action_names),
        ", ".join(action_names),
    )
    return action_spec, action_normalizers, action_names

  def _get_observation_spec(
      self, devices: Sequence[DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[str]]:
    """Constructs the observation specification for the environment.

    The observation space is a flat array of float32 values. It includes:
    - Sensor readings from devices (potentially processed by a histogram reducer).
    - Auxiliary features like time of day, day of week, and occupancy.

    This method delegates to either
    `_get_observation_spec_single_timeseries` or
    `_get_observation_spec_histogram_reducer` based on whether an
    `observation_histogram_reducer` is configured.

    Args:
      devices: A sequence of `DeviceInfo` protocol buffers describing the
        available devices and their observable fields.

    Returns:
      A tuple containing:
        - obs_spec: An `tf_agents.specs.array_spec.ArraySpec` defining the
          shape and dtype (float32) of the observation space.
        - observable_fields: A sequence of `DeviceFieldId` strings (or
          generated histogram bin IDs and auxiliary feature labels)
          indicating the order of observations in the `obs_spec` array.
    """
    # TODO(sipple): Desuplicate the else case of
    # _get_observation_spec_histogram_reducer if the same as
    # _get_observation_spec_single_timeseries.

    if self._observation_histogram_reducer is None:
      obs_spec, observable_fields = (
          self._get_observation_spec_single_timeseries(devices)
      )
    else:
      obs_spec, observable_fields = (
          self._get_observation_spec_histogram_reducer(devices)
      )

    logging.info("There are %d observable fields.", len(observable_fields))
    logging.info("observable_fields: %s", observable_fields)
    return obs_spec, observable_fields

  def _get_observation_spec_histogram_reducer(
      self, devices: Sequence[DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[str]]:
    """Builds observation spec when using a histogram reducer.

    If a field is configured for histogram reduction, its observation
    will be a set of bin counts. Other fields are passed through directly.
    Auxiliary time and occupancy features are also appended.

    Args:
      devices: Sequence of `DeviceInfo` protos.

    Returns:
      A tuple (obs_spec, observable_fields_with_bins_and_aux).
        - obs_spec: `ArraySpec` for the observation vector.
        - observable_fields_with_bins_and_aux: Ordered list of field names,
          including generated bin names for histogrammed features (e.g.,
          'measurement_name_h_0.50') and auxiliary feature names.
    """
    assert self._observation_histogram_reducer is not None

    observable_fields = []

    for device in sorted(devices, key=lambda x: x.device_id):
      for measurement_name in sorted(device.observable_fields):
        device_id = DeviceId(device.device_id)
        measurement_name = FieldName(measurement_name)
        if (
            measurement_name
            in self._observation_histogram_reducer.histogram_parameters.keys()
        ):
          for v in self._observation_histogram_reducer.histogram_parameters[
              measurement_name
          ]:
            bin_id = f"h_{v:.2f}"
            if (measurement_name, bin_id) not in self._id_map.keys():
              field_id = DeviceFieldId(f"{measurement_name}_{bin_id}")

              self._id_map[(measurement_name, bin_id)] = field_id
              logging.info(
                  "Histogram feature: %s %s added to the id_map.",
                  measurement_name,
                  bin_id,
              )
              observable_fields.append(field_id)

        else:
          field_id = generate_field_id(
              device_id, measurement_name, self._id_map
          )
          self._id_map[(device_id, measurement_name)] = field_id
          logging.info(
              "Passthrough feature: %s %s",
              device_id,
              measurement_name,
          )
          observable_fields.append(field_id)

    # Include the temporal features.
    observable_fields.extend(self._auxiliary_features)

    obs_spec = array_spec.ArraySpec(
        shape=(len(observable_fields),), dtype=np.float32, name="observation"
    )
    return obs_spec, observable_fields

  def _get_observation_spec_single_timeseries(
      self, devices: Sequence[DeviceInfo]
  ) -> tuple[types.ArraySpec, Sequence[str]]:
    """Builds observation spec for single timeseries data (no histogram).

    Each observable field from each device directly becomes an element in the
    observation vector. Auxiliary time and occupancy features are appended.

    Args:
      devices: Sequence of `DeviceInfo` protos.

    Returns:
      A tuple (obs_spec, observable_fields_and_aux):
        - obs_spec: `ArraySpec` for the observation vector.
        - observable_fields_and_aux: Ordered list of `DeviceFieldId`s and
          auxiliary feature names.
    """
    observable_fields = []
    for device in sorted(devices, key=lambda x: x.device_id):
      for measurement_name in sorted(device.observable_fields):
        device_id = DeviceId(device.device_id)
        measurement_name = FieldName(measurement_name)

        field_id = generate_field_id(device_id, measurement_name, self._id_map)
        self._id_map[(device_id, measurement_name)] = field_id
        observable_fields.append(field_id)

    # Include the temporal features.
    observable_fields.extend(self._auxiliary_features)

    # Multiple attempts to use a map of field_name:values for
    # the observation spec failed in various locations, including
    # (a) the ActorDistributionNetwork with various combinations
    # of preprocessing combiners, and (b) the replay buffer when adding
    # trajectories. By mapping to a simple flat ArraySpec, the failures
    # were reliably prevented and allowed the agent to train.

    logging.info("There are %d observable fields.", len(observable_fields))

    obs_spec = array_spec.ArraySpec(
        shape=(len(observable_fields),), dtype=np.float32, name="observation"
    )
    return obs_spec, observable_fields

  @property
  def current_simulation_timestamp(self) -> pd.Timestamp:
    """The current timestamp within the building simulation."""
    return self.building.current_timestamp

  def _get_action_value_type(self, field_id: DeviceFieldId) -> ValueType:
    """Determines the `ValueType` of an action or observation field.

    Note: This method seems to primarily infer based on Python types,
    which might not always align with the `ValueType` enum if not carefully managed.
    It checks if the `field_id` is in `_action_names` to look up its spec,
    otherwise assumes it's an observation.

    Args:
      field_id: The `DeviceFieldId` of the field to check.

    Returns:
      The `ValueType` (e.g., `VALUE_INTEGER`, `VALUE_BINARY`, `VALUE_CONTINUOUS`)
      associated with the field's data type. Returns `VALUE_TYPE_UNDEFINED`
      if the type cannot be determined or is not supported.
    """
    if field_id in self._action_names:
      # This indexing might fail if action_spec() is a BoundedArraySpec
      # and not a dict of specs. This part of the code might need review
      # based on the actual structure of self.action_spec().
      # Assuming self.action_spec() is a flat BoundedArraySpec,
      # individual field specs are not directly queryable by field_id here.
      # However, the original code implies it might be a dict-like spec
      # before being converted to a BoundedArraySpec, or this method
      # is called when it's still in a dict-like form.
      # For now, we'll assume the logic reflects the intended use.
      spec = self.action_spec()[field_id] # type: ignore
    else:
      spec = self.observation_spec()[field_id] # type: ignore

    if spec.dtype == array_spec.ArraySpec((), int): # type: ignore
      return ValueType.VALUE_INTEGER
    if spec.dtype == array_spec.ArraySpec((), bool): # type: ignore
      return ValueType.VALUE_BINARY
    if spec.dtype == array_spec.ArraySpec((), np.float32): # type: ignore
      return ValueType.VALUE_CONTINUOUS
    # categorical not supported
    return ValueType.VALUE_TYPE_UNDEFINED

  def _create_action_request(self, action_array: np.ndarray) -> ActionRequest:
    """Converts a normalized agent action array into an `ActionRequest`.

    The `action_array` from the agent (typically normalized values between -1 and 1)
    is denormalized to native engineering units using the configured
    `action_normalizers`.

    Args:
      action_array: A NumPy array representing the agent's action, with
        values corresponding to `self._action_names` in order.

    Returns:
      An `ActionRequest` proto message ready to be sent to the building
      simulation, containing denormalized setpoint values.
    """
    timestamp = conversion_utils.pandas_to_proto_timestamp(
        self.building.current_timestamp
    )
    else:
      spec = self.observation_spec()[field_id]

    if spec.dtype == array_spec.ArraySpec((), int):
      return ValueType.VALUE_INTEGER
    if spec.dtype == array_spec.ArraySpec((), bool):
      return ValueType.VALUE_BINARY
    if spec.dtype == array_spec.ArraySpec((), np.float32):
      return ValueType.VALUE_CONTINUOUS
    # categorical not supported
    return ValueType.VALUE_TYPE_UNDEFINED

    action_request = ActionRequest(timestamp=timestamp)

    action_map = {}
    for i in range(len(self._action_names)):
      action_map[self._action_names[i]] = action_array[i]

    # Append the action to the action history for use in computing cost/penalty
    # for large changes in the action.
    self._action_history.append(
        np.array(np.fromiter(action_map.values(), dtype=np.float32))
    )

    for field_id, agent_action_value in action_map.items():
      device_id, setpoint_name = self._id_map.inv[field_id]
      action_normalizer = self.action_normalizers[field_id]
      native_action_value = action_normalizer.setpoint_value(
          agent_action_value
      )

      single_action_request = SingleActionRequest(
          device_id=device_id,
          setpoint_name=setpoint_name,
          continuous_value=native_action_value,
      )
      action_request.single_action_requests.append(single_action_request)

    return action_request

  def _get_observation(self) -> np.ndarray:
    """Retrieves, processes, and returns the current environment observation.

    This involves:
    1. Requesting raw observations from the building simulation.
    2. Filling any missing values using the previous observation.
    3. Optionally writing raw and image-based observations if a metrics writer
       is configured.
    4. Normalizing the observations.
    5. Converting the normalized `ObservationResponse` into a flat NumPy array,
       potentially using a histogram reducer.
    6. Appending auxiliary features (time, comfort, occupancy).
    7. Logging any NaNs or Infs found in the final observation vector.

    Returns:
      A 1D NumPy array of float32 values representing the fully processed
      observation vector, ordered according to `self.field_names`.

    Raises:
      ValueError: If the final observation vector has a different number of
        elements than expected by `self.field_names`.
    """
    timestamp = conversion_utils.pandas_to_proto_timestamp(
        self.building.current_timestamp
    )
    observation_request = ObservationRequest()
    observation_request.CopyFrom(self._observation_request)
    observation_request.timestamp.CopyFrom(timestamp)

    observation_response = self.building.request_observations(
        observation_request
    )

    observation_response = replace_missing_observations_past(
        current_observation_response=observation_response,
        past_observation_response=self._last_observation_response,
    )
    self._last_observation_response = observation_response

    if self._metrics_writer:
      self._metrics_writer.write_observation_response(
          observation_response, self.current_simulation_timestamp
      )
      if self._building_image_generator:
        building_image = self._building_image_generator.generate_building_image(
            observation_response
        )
        self._metrics_writer.write_building_image(
            building_image, self.current_simulation_timestamp
        )

    normalized_observation_response = self._observation_normalizer.normalize(
        observation_response
    )

    if self._observation_histogram_reducer is None:
      observation = self._normalized_observation_response_to_observation_map_single_timeseries(  # pylint: disable=line-too-long
          normalized_observation_response
      )
    else:
      observation = self._normalized_observation_response_to_observation_map_histogram_reducer(  # pylint: disable=line-too-long
          normalized_observation_response
      )

    hod_rad = conversion_utils.get_radian_time(
        self.current_simulation_timestamp,
        conversion_utils.TimeIntervalEnum.HOUR_OF_DAY,
    )

    hod_features = regression_building_utils.expand_time_features(
        self._num_hod_features, hod_rad, HOD_LABEL
    )
    for hod_feature_name in hod_features:
      observation[f"{hod_feature_name[0]}_{hod_feature_name[1]}"] = np.array(
          hod_features[hod_feature_name], dtype=np.float32
      )

    dow_rad = conversion_utils.get_radian_time(
        self.current_simulation_timestamp,
        conversion_utils.TimeIntervalEnum.DAY_OF_WEEK,
    )

    dow_features = regression_building_utils.expand_time_features(
        self._num_dow_features, dow_rad, DOW_LABEL
    )
    for dow_feature_name in dow_features:
      observation[f"{dow_feature_name[0]}_{dow_feature_name[1]}"] = np.array(
          dow_features[dow_feature_name], dtype=np.float32
      )

    observation[COMFORT_MODE_NOW] = np.array(
        self.building.is_comfort_mode(self.current_simulation_timestamp),
        dtype=np.float32,
    )
    observation[COMFORT_MODE_SOON] = np.array(
        self.building.is_comfort_mode(
            self.current_simulation_timestamp + pd.Timedelta(60, unit="minute")
        ),
        dtype=np.float32,
    )
    observation[NUM_OCCUPANTS] = np.array(
        (self.building.num_occupants - self.occupancy_normalization_constant)
        / (self.occupancy_normalization_constant + 1),
        dtype=np.float32,
    )
    # Return observation as a flat array.
    if len(self.field_names) > len(observation):
      dif_set = set(self.field_names) - observation.keys()
      dif_set_str = ", ".join(dif_set)
      logging.error("Difference: %s", dif_set_str)
      raise ValueError(
          f"Observation of length ({len(observation)}) is missing"
          f" {len(dif_set)} fields from expected fields size"
          f" ({len(self.field_names)})."
      )

    obsarray = np.array(
        [observation[field_id] for field_id in self.field_names],
        dtype=np.float32,
    )
    nan_ix = np.squeeze(np.argwhere(np.isnan(obsarray)), axis=1)
    if nan_ix.size > 0:
      nan_fields = [self.field_names[i] for i in nan_ix]
      logging.warning(
          "Observation vector contains Nans at %s.", ", ".join(nan_fields)
      )
    inf_ix = np.squeeze(np.argwhere(np.isinf(obsarray)), axis=1)
    # TODO(sipple) Add a unit test for the logging below.
    if inf_ix.size > 0:
      inf_fields = [self.field_names[i] for i in inf_ix]
      logging.warning(
          "Observation vector contains Infs at %s.", ", ".join(inf_fields)
      )
    return obsarray

  def _normalized_observation_response_to_observation_map_single_timeseries(
      self,
      normalized_observation_response: ObservationResponse,
  ) -> dict[str, np.ndarray]:
    """Converts a normalized ObservationResponse to a feature map (no histogram).

    Each valid, continuous measurement in the `normalized_observation_response`
    is extracted and placed into a dictionary, keyed by its `DeviceFieldId`.
    This method is used when no `observation_histogram_reducer` is active.

    Args:
      normalized_observation_response: An `ObservationResponse` proto message
        where the observation values have already been normalized.

    Returns:
      A dictionary mapping `DeviceFieldId` strings to 1D NumPy arrays
      (each containing a single float32 value) representing the normalized
      observation for that field. Invalid observations are logged and skipped.
    """
    observation_map = {}
    responses = normalized_observation_response.single_observation_responses
    for single_observation_response in responses:
      request = single_observation_response.single_observation_request
      device_id = request.device_id
      measurement_name = request.measurement_name
      continuous_value = single_observation_response.continuous_value

      if not single_observation_response.observation_valid:
        logging.warn(
            "Invalid observation reported %s %s %f",
            device_id,
            measurement_name,
            continuous_value,
        )
        continue

      field_id = self._id_map[(device_id, measurement_name)]

      value = np.array(
          single_observation_response.continuous_value, dtype=np.float32
      )

      observation_map[field_id] = value
    return observation_map

  def _normalized_observation_response_to_observation_map_histogram_reducer(
      self,
      normalized_observation_response: ObservationResponse,
  ) -> dict[str, np.ndarray]:
    """Converts a normalized ObservationResponse to a feature map using histogram reduction.

    This method processes the `normalized_observation_response` using the
    configured `_observation_histogram_reducer`. The reducer transforms
    sequences of observations (potentially multiple time steps, though here
    applied to a single response) into a reduced feature set, often involving
    binning values into histograms.

    Args:
      normalized_observation_response: An `ObservationResponse` proto message
        with normalized values.

    Returns:
      A dictionary where keys are feature names (which may include original
      field names or names generated by the histogram reducer for bins) and
      values are the corresponding reduced feature values (typically NumPy
      arrays or scalars).
    """
    assert self._observation_histogram_reducer is not None

    feature_tuples = regression_building_utils.get_feature_tuples(
        normalized_observation_response
    )

    observation_sequence = regression_building_utils.get_observation_sequence(
        [normalized_observation_response],
        feature_tuples,
        self._time_zone,
        self._num_hod_features,
        self._num_dow_features,
    )
    rs = self._observation_histogram_reducer.reduce(
        observation_sequence
    ).reduced_sequence

    observation_map = rs.iloc[0].to_dict()
    observation_map = {
        "_".join(k): observation_map[k]
        for k in observation_map
        if isinstance(k, tuple)
    }
    return observation_map

  def _get_reward(self) -> float:
    """Calculates the reward for the current step.

    1. Retrieves `RewardInfo` from the building (which contains raw metrics
       like energy use, comfort violations, etc.).
    2. Computes the scalar reward value using the configured `reward_function`.
    3. Writes detailed reward metrics to file and TensorBoard summaries if
       writers are configured.

    Returns:
      The scalar reward value for the agent at the current step.
    """
    # Get the reward input (RewardInfo) from the building.
    reward_info = self.building.reward_info
    # Using the reward function, compute the reward value.
    reward_response = self.reward_function.compute_reward(reward_info)

    # Write both RewardInfo and RewardResponse if a metrics writer is
    # enabled.
    if self._metrics_writer:
      self._metrics_writer.write_reward_info(
          reward_info, self.current_simulation_timestamp
      )
      self._metrics_writer.write_reward_response(
          reward_response, self.current_simulation_timestamp
      )

    # Summary writer commits additional metrics to TensorBoard.
    if self._summary_writer:
      self._write_summary_reward_info_metrics(reward_info)
      self._write_summary_reward_response_metrics(reward_response)
      self._commit_reward_metrics()

    return reward_response.agent_reward_value

  def _write_summary_reward_info_metrics(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> None:
    """Accumulates components of `RewardInfo` for TensorBoard logging.

    Extracts specific metrics from the `RewardInfo` protobuf (e.g., different
    types of energy consumption) and appends them to internal accumulators.
    These accumulated values will later be averaged and written to TensorBoard
    by `_commit_reward_metrics`.

    Args:
      reward_info: The `RewardInfo` protobuf message received from the building
        simulation for the current step.
    """
    energy_use = conversion_utils.get_reward_info_energy_use(reward_info)

    self._accumulator["electrical_energy"].append(
        energy_use["air_handler_blower_electricity"]
        + energy_use["air_handler_air_conditioning"]
        + energy_use["boiler_pump_electrical_energy"]
    )
    self._accumulator["natural_gas_energy"].append(
        energy_use["boiler_natural_gas_heating_energy"]
    )

  def _write_summary_reward_response_metrics(
      self, reward_response: smart_control_reward_pb2.RewardResponse
  ) -> None:
    """Accumulates components of `RewardResponse` for TensorBoard logging.

    Extracts specific metrics from the `RewardResponse` protobuf (e.g., costs,
    carbon emissions, regret) and appends them to internal accumulators.
    These will be averaged and written to TensorBoard by `_commit_reward_metrics`.

    Args:
      reward_response: The `RewardResponse` protobuf message generated by the
        `reward_function` for the current step.
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
    self._accumulator["step_duration_sec"].append(
        reward_response.normalized_productivity_regret
    )

  def _commit_reward_metrics(self) -> None:
    """Writes accumulated reward metrics to TensorBoard and resets accumulators.

    This method is called periodically (controlled by
    `_metrics_reporting_interval`). It calculates the mean of all metrics
    accumulated since its last execution (e.g., various energy costs, carbon
    emissions, productivity regret) and writes these mean values as scalars
    to the TensorBoard summary logs. After writing, it clears the accumulators.

    Requires `self._summary_writer` to be configured.
    """
    assert self._summary_writer is not None

    if self._global_step_count % self._metrics_reporting_interval == 0:
      with (  # pylint: disable=not-context-manager # TF summary ops are context managers
          self._summary_writer.as_default(),
          tf.compat.v2.summary.record_if(True),
          tf.name_scope("RewardInfo/"), # Groups reward metrics in TensorBoard
      ):
        for key, values in self._accumulator.items():
          if values: # Ensure there's data to average
            tf.compat.v2.summary.scalar(
                name=key,
                data=np.mean(values), # Log the mean of accumulated values
                step=self._global_step_count,
            )
        # Reset accumulator for the next reporting interval
        self._accumulator = collections.defaultdict(list)

  @property
  def label(self) -> str:
    """A label for the environment instance, used in logging and metrics."""
    return self._label

  def _reset(self) -> ts.TimeStep:
    """Resets the environment to the beginning of a new episode.

    This involves:
    1. Resetting the underlying building simulation.
    2. Clearing internal accumulators and action history.
    3. Incrementing the episode counter.
    4. Setting up metric writers for the new episode if `metrics_path` is configured.
    5. Writing initial device and zone information to metrics if applicable.
    6. Recalculating episode start and end timestamps.
    7. Obtaining the initial observation from the building.

    Returns:
      A `tf_agents.trajectories.time_step.TimeStep` object representing the
      initial state of the environment (type `FIRST`).
    """
    self.building.reset()

    self._accumulator = collections.defaultdict(list)

    self._episode_ended = False
    self._episode_count += 1
    self._episode_cumulative_reward = 0

    observation = self._get_observation()
    self._action_history = []

    now = pd.Timestamp.utcnow()

    self._metrics_writer = None

    if self.metrics_path and self._writer_factory:
      episode_metrics_id = f"{self._label}_{now:%y%m%d_%H%M%S}"
      output_dir = os.path.join(self.metrics_path, episode_metrics_id)

      logging.info("Writing metric files to %s", output_dir)
      self._metrics_writer = self._writer_factory.create(output_dir)

      if self._building_image_generator:
        img_file_path = os.path.join(
            output_dir, constants.BUILDING_IMAGE_CSV_FILE
        )
        logging.info("Writing building image files to %s", img_file_path)

    if self._metrics_writer:
      logging.info("Writing %d device_infos.", len(self.building.devices))
      self._metrics_writer.write_device_infos(self.building.devices)
      logging.info("Writing %d zone_infos.", len(self.building.zones))
      self._metrics_writer.write_zone_infos(self.building.zones)

    self._episode_start_time = time.time()
    self._step_count = 0
    self._start_timestamp = self.building.current_timestamp
    self._end_timestamp = (
        self._start_timestamp
        + self._num_timesteps_in_episode * self._step_interval
    )
    logging.info(
        "Restarting the environment for %s to %s",
        self._start_timestamp,
        self._end_timestamp,
    )
    return ts.restart(observation)

  @gin.configurable
  def action_spec(self) -> types.NestedArraySpec:
    return self._action_spec

  @gin.configurable
  def observation_spec(self) -> types.NestedArraySpec:
    return self._observation_spec

  def _format_action(
      self, action: types.NestedArray, action_names: Sequence[str]
  ) -> types.NestedArray:
    """Allows derived classes to reformat actions before processing.

    This method acts as a hook for subclasses of `Environment` that might
    use a different action representation internally than the base environment.
    By overriding this method, those subclasses can convert their specific
    action format into the standard flat array format expected by
    `_create_action_request`.

    In this base class, this method is a no-op and returns the action as is.
    The `action_names` argument is provided for context in overridden methods
    but is unused here, hence `pylint: disable=unused-argument` might be
    appropriate if not used by any subclass that calls super().

    Args:
      action: The action(s) to be potentially formatted. This is typically a
        NumPy array or a nested structure of arrays.
      action_names: A sequence of strings representing the names of the actions,
        corresponding to the elements in the `action` array if it's flat.

    Returns:
      The action, possibly reformatted. In this base implementation, it's the
      original `action` unchanged.
    """
    # pylint: disable=unused-argument
    return action

  def _step(self, action: types.NestedArray) -> ts.TimeStep:
    """Processes a single timestep in the environment.
    
    This method is called by the TF-Agents framework to advance the simulation
    by one step. It performs the following operations:
    1. Checks if the episode has ended; if so, resets the environment.
    2. Formats the incoming `action` using `_format_action` (a hook for subclasses).
    3. Converts the (potentially formatted and normalized) `action` array into
       an `ActionRequest` protobuf message with native engineering values.
    4. Sends the `ActionRequest` to the building simulation.
    5. Handles the `ActionResponse` from the building, checking if actions were
       accepted. If actions are rejected (e.g., due to a `RuntimeError` from
       the building model or explicit rejection in the response), a special
       negative infinity reward (`ACTION_REJECTION_REWARD`) can be assigned.
    6. Writes the `ActionResponse` to metrics if a writer is configured.
    7. Advances the building simulation time (`building.wait_time()`).
    8. Retrieves the new observation from the building using `_get_observation()`.
    9. Calculates the reward for the step using `_get_reward()`. If actions
       were rejected, this reward might be overridden.
    10. Checks if the episode has ended based on the new state or step count.
    11. Logs step and episode statistics.
    12. Returns a `TimeStep` object (either `TERMINATION` or `TRANSITION`)
        containing the new observation, reward, discount factor, and step type.

    Args:
      action: A `types.NestedArray` (typically a NumPy array) representing the
        action taken by the agent. This action is usually normalized.

    Returns:
      A `tf_agents.trajectories.time_step.TimeStep` object representing the
      result of the action, containing the new observation, reward, and whether
      the episode has terminated or is ongoing.
    """

    def _action_strings(
        action_request: ActionRequest,
    ) -> Sequence[str]:
      """Create a list of actions from an ActionRequest for logging."""
      action_strings = []
      for single_action_request in action_request.single_action_requests:
        action_string = (
            f"{single_action_request.device_id} "
            f"{single_action_request.setpoint_name}: "
            f"{single_action_request.continuous_value:3.2f}"
        )
        action_strings.append(action_string)
      return action_strings

    if self._episode_ended:
      return self.reset()

    t0 = time.time()
    reward_value = 0.0
    observation = None

    # Reformat actions if necessary.
    action = self._format_action(action, self._action_names)

    # Convert the action from normalized to native values.
    action_request = self._create_action_request(action)

    try:
      # Send the action request to the building.
      action_response = self.building.request_action(action_request)

    except RuntimeError as err:
      # If the building rejects the request, create an action response
      # indicating that the request was rejected.
      action_accepted = False

      action_response = _apply_action_response(
          action_request,
          response_timestamp=self.current_simulation_timestamp,
          action_response_type=SingleActionResponse.ActionResponseType.REJECTED_NOT_ENABLED_OR_AVAILABLE,  # pylint: disable=line-too-long
          additional_info=str(err),
      )
      logging.exception(
          "Action REJECTED at %s: %s.",
          self.current_simulation_timestamp,
          ", ".join(_action_strings(action_request)),
      )

    else:
      action_accepted = all_actions_accepted(action_response)

    if self._metrics_writer and action_response is not None:
      self._metrics_writer.write_action_response(
          action_response, self.current_simulation_timestamp
      )

    self.building.wait_time()

    observation = self._get_observation()

    # We need to signal to the Actor that action was rejected and not to
    # append this observation/action request to the trajectory.
    # Since TimeStep cannot be extended and it is checked for NaNs,
    # we apply -inf as a reward to indicate the rejection.
    # This requires a specialized Actor extension class to handle the
    # rejection.
    reward_value = self._get_reward()
    if not action_accepted:
      reward_value = ACTION_REJECTION_REWARD

    # Exit when the episode has ended and return terminal step information.
    # We still need to get the final observation to add to the transition.
    self._episode_ended = self._has_episode_ended()

    self._episode_cumulative_reward += reward_value

    t1 = time.time()
    episode_dt = t1 - self._episode_start_time
    step_dt = t1 - t0

    if self._episode_ended:
      logging.info(
          "%s: Terminating episode=%d step=%d current_time=%s step_reward=%4.2f"
          " cumulative_reward=%5.2f episode_time=%5.2fs step_time=%3.2fs",
          self._label,
          self._episode_count,
          self._step_count,
          self.building.current_timestamp,
          reward_value,
          self._episode_cumulative_reward,
          episode_dt,
          step_dt,
      )
      termination = ts.termination(observation, reward_value)
      return termination

    else:
      transition = ts.transition(
          observation, reward_value, self.discount_factor
      )

      if self._step_count % 100 == 0:
        logging.info(
            (
                "%s: episode=%d step=%d current_time=%s step_reward=%4.2f"
                " cumulative_reward=%5.2f episode_time=%5.2fs step_time=%3.2fs"
            ),
            self._label,
            self._episode_count,
            self._step_count,
            self.building.current_timestamp,
            reward_value,
            self._episode_cumulative_reward,
            episode_dt,
            step_dt,
        )

      self._step_count += 1
      self._global_step_count += 1
      return transition

  def render(self, mode: str = "rgb_array") -> Optional[types.NestedArray]:
    """Renders the environment. Not currently implemented.

    Args:
      mode: The rendering mode (e.g., "rgb_array").

    Returns:
      An optional NumPy array representing the rendered image, or None.

    Raises:
      NotImplementedError: This method is not yet implemented.
    """
    raise NotImplementedError("Rendering not supported yet.")

  def _has_episode_ended(self) -> bool:
    """Checks if the current episode has reached its maximum duration.

    Returns:
      True if the current `_step_count` is greater than or equal to
      `_num_timesteps_in_episode`, False otherwise.
    """
    return self._step_count >= self._num_timesteps_in_episode


def _apply_action_response(
    action_request: ActionRequest,
    action_response_type: SingleActionResponse.ActionResponseType,
    response_timestamp: pd.Timestamp,
    additional_info: Optional[str] = None,
) -> ActionResponse:
  """Constructs an `ActionResponse` for a given `ActionRequest`.

  This helper function is typically used when the building simulation itself
  doesn't generate an `ActionResponse (e.g., in case of an internal error
  before the request is processed by the building). It creates an
  `ActionResponse` where each `SingleActionResponse` reflects the provided
  `action_response_type`.

  Args:
    action_request: The `ActionRequest` for which to generate a response.
    action_response_type: The `ActionResponseType` (e.g., REJECTED, ACCEPTED)
      to be set for all single action responses.
    response_timestamp: The timestamp for the generated `ActionResponse`.
    additional_info: Optional string providing more details about the response,
      to be included in each `SingleActionResponse`.

  Returns:
    An `ActionResponse` protobuf message.
  """
  single_action_responses = [
      _apply_single_action_response(
          single_action_request, action_response_type, additional_info
      )
      for single_action_request in action_request.single_action_requests
  ]
  return ActionResponse(
      timestamp=conversion_utils.pandas_to_proto_timestamp(response_timestamp),
      request=action_request,
      single_action_responses=single_action_responses,
  )


def _apply_single_action_response(
    single_action_request: SingleActionRequest,
    action_response_type: SingleActionResponse.ActionResponseType,
    additional_info: Optional[str] = None,
) -> SingleActionResponse:
  """Constructs a `SingleActionResponse` for a `SingleActionRequest`.

  This is a helper for `_apply_action_response` to create individual
  responses for each action within an `ActionRequest`.

  Args:
    single_action_request: The `SingleActionRequest` for which to generate a
      response.
    action_response_type: The `ActionResponseType` to set.
    additional_info: Optional string with more details.

  Returns:
    A `SingleActionResponse` protobuf message.
  """
  return SingleActionResponse(
      request=single_action_request,
      response_type=action_response_type,
      additional_info=additional_info,
  )
