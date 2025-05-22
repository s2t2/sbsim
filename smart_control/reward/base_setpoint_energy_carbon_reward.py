"""Defines an abstract base class for reward functions in smart buildings.

This module provides `BaseSetpointEnergyCarbonRewardFunction`, an abstract
base class intended for reward functions that calculate a scalar reward signal
based on a combination of:
- Estimated occupant productivity (derived from thermal comfort, i.e., how close
  zone temperatures are to their setpoints).
- Energy consumption costs.
- Carbon emissions impact.

Concrete implementations are expected to combine these factors into a final
reward value. This base class provides common helper methods for calculating
productivity and summing energy rates from `RewardInfo` protobuf messages.
"""

from typing import Tuple

import gin
import numpy as np

from smart_control.models.base_reward_function import BaseRewardFunction
from smart_control.proto import smart_control_reward_pb2 # For type hinting
from smart_control.utils import conversion_utils


@gin.configurable()
class BaseSetpointEnergyCarbonRewardFunction(BaseRewardFunction):
  """Abstract base for reward functions considering productivity, energy, and carbon.

  This class provides foundational methods for calculating occupant productivity
  based on thermal comfort (deviation from setpoints) and for summing energy
  consumption rates. Concrete subclasses must implement `compute_reward` to
  define how these components, along with energy costs and carbon emissions
  (obtained via energy cost models), are combined into a final scalar reward
  and a `RewardResponse` protobuf.

  The productivity model assumes that productivity is maximal when the zone
  temperature is within the heating and cooling setpoint deadband. Outside this
  deadband, productivity decreases following a sigmoid decay curve.
  """

  @gin.configurable()
  def __init__(
      self,
      max_productivity_personhour_usd: float,
      productivity_midpoint_delta: float,
      productivity_decay_stiffness: float,
  ):
    """Initializes the base reward function with productivity parameters.

    Args:
      max_productivity_personhour_usd: The maximum assumed productivity value
        (e.g., in USD) per person, per hour, when thermal conditions are
        optimal (i.e., zone temperature is within the setpoint deadband).
      productivity_midpoint_delta: The temperature difference (in the same units
        as setpoints and zone temperatures, typically Kelvin) from the heating
        or cooling setpoint at which the estimated productivity drops to 50% of
        `max_productivity_personhour_usd`. This defines the center of the
        sigmoid decay curve for productivity loss.
      productivity_decay_stiffness: A parameter controlling the steepness of the
        sigmoid curve used to model the decline in productivity as temperature
        deviates further from the setpoint deadband. Higher values result in a
        sharper drop in productivity.
    """
    self._max_productivity_personhour_usd = max_productivity_personhour_usd
    self._productivity_midpoint_delta = productivity_midpoint_delta
    self._productivity_decay_stiffness = productivity_decay_stiffness

  @abc.abstractmethod # Mark as abstract as per original class structure
  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Computes the overall reward based on building state and performance.

    Concrete subclasses must implement this method. The implementation should
    utilize the helper methods provided by this base class (e.g., for
    productivity, energy rates) and integrate them with energy cost and carbon
    emission calculations (likely from `BaseEnergyCost` model instances) to
    produce a `RewardResponse`. The `RewardResponse` includes the final scalar
    `agent_reward_value` and can also store disaggregated components of the
    reward for analysis.

    Args:
      reward_info: A `smart_control_reward_pb2.RewardInfo` protobuf message
        containing detailed information about the building's state over the last
        control interval, including zone temperatures, setpoints, occupancy,
        and energy consumption rates by various devices.

    Returns:
      A `smart_control_reward_pb2.RewardResponse` protobuf message containing
      the calculated `agent_reward_value` and other relevant reward components.
    """
    raise NotImplementedError("Subclasses must implement compute_reward.")

  def _sum_zone_productivities(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> Tuple[float, float]:
    """Calculates total estimated productivity and occupancy across all zones.

    This method iterates through all zones defined in `energy_reward_info`,
    calculates the productivity for each zone using
    `_get_zone_productivity_reward`, and sums these values. It also sums the
    average occupancy across all zones.

    Args:
      energy_reward_info: A `smart_control_reward_pb2.RewardInfo` protobuf
        containing zone-specific data like setpoints, temperatures, and
        occupancy.

    Returns:
      A tuple `(cumulative_productivity_value, total_average_occupancy)`:
        - cumulative_productivity_value: The sum of estimated productivity
          values (e.g., in USD) across all zones for the reward interval.
        - total_average_occupancy: The sum of average occupancy values across
          all zones.
    """
    time_interval_sec = self._get_delta_time_sec(energy_reward_info)
    cumulative_productivity = 0.0
    total_occupancy = 0.0

    for zid, zone_info in energy_reward_info.zone_reward_infos.items(): # pytype: disable=attribute-error
      occupancy = zone_info.average_occupancy
      total_occupancy += occupancy
      cumulative_productivity += self._get_zone_productivity_reward(
          heating_setpoint=zone_info.heating_setpoint_temperature,
          cooling_setpoint=zone_info.cooling_setpoint_temperature,
          zone_temp=zone_info.zone_air_temperature,
          time_interval_sec=time_interval_sec,
          average_occupancy=occupancy,
      )
    return cumulative_productivity, total_occupancy

  def _get_zone_productivity_reward(
      self,
      heating_setpoint: float,
      cooling_setpoint: float, # Added type hint
      zone_temp: float,
      time_interval_sec: float,
      average_occupancy: float, # Added type hint
  ) -> float:
    """Computes estimated productivity for a zone based on thermal comfort.

    Productivity is assumed to be at its maximum (`_max_productivity_personhour_usd`)
    when `zone_temp` is between `heating_setpoint` and `cooling_setpoint`.
    If `zone_temp` falls below `heating_setpoint` or rises above
    `cooling_setpoint`, productivity decreases following a sigmoid function.
    The shape of this decay is determined by `_productivity_midpoint_delta`
    and `_productivity_decay_stiffness`.

    The final productivity value is scaled by `average_occupancy` and the
    duration of the `time_interval_sec` (converted to hours).

    Args:
      heating_setpoint: The heating setpoint temperature for the zone.
      cooling_setpoint: The cooling setpoint temperature for the zone.
      zone_temp: The actual average air temperature in the zone.
      time_interval_sec: The duration of the reward interval in seconds.
      average_occupancy: The average number of occupants in the zone during
        the interval.

    Returns:
      The estimated productivity value (e.g., in USD) for the zone over the
      given time interval, considering the occupancy.
    """
    # Midpoint of the sigmoid decay curve for temperatures below heating setpoint
    x0_low = heating_setpoint - self._productivity_midpoint_delta
    # Midpoint of the sigmoid decay curve for temperatures above cooling setpoint
    x0_high = cooling_setpoint + self._productivity_midpoint_delta

    if zone_temp < heating_setpoint:
      # Productivity decays as temperature drops further below heating setpoint
      productivity_per_personhour = self._max_productivity_personhour_usd / (
          1.0 + np.exp(-self._productivity_decay_stiffness * (zone_temp - x0_low))
      )
    elif zone_temp > cooling_setpoint:
      # Productivity decays as temperature rises further above cooling setpoint
      productivity_per_personhour = self._max_productivity_personhour_usd * (
          1.0 - 1.0 / (
              1.0 + np.exp(-self._productivity_decay_stiffness * (zone_temp - x0_high))
          )
      )
    else:
      # Temperature is within the deadband; productivity is maximal
      productivity_per_personhour = self._max_productivity_personhour_usd

    # Scale productivity by occupancy and duration (converted from seconds to hours)
    return productivity_per_personhour * average_occupancy * (time_interval_sec / 3600.0)

  def _get_delta_time_sec(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Calculates the duration of the reward interval in seconds.

    The duration is determined from the `start_timestamp` and `end_timestamp`
    fields within the `energy_reward_info` protobuf.

    Args:
      energy_reward_info: A `smart_control_reward_pb2.RewardInfo` protobuf.

    Returns:
      The duration of the reward interval in seconds (float).
    """
    start_time = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.start_timestamp # pytype: disable=attribute-error
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(
        energy_reward_info.end_timestamp # pytype: disable=attribute-error
    )
    return (end_time - start_time).total_seconds()

  def _sum_electricity_energy_rate(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Calculates the total electrical energy consumption rate (in Watts).

    This sum includes:
    - Blower electrical energy rate for all air handlers.
    - Absolute value of air conditioning electrical energy rate for all air
      handlers (to account for both heating and cooling energy as positive consumption).
    - Pump electrical energy rate for all boilers.

    Args:
      energy_reward_info: A `smart_control_reward_pb2.RewardInfo` protobuf
        containing per-device energy consumption rates.

    Returns:
      The total electrical power consumption rate in Watts (float).
    """
    electrical_energy_rate = 0.0
    for ahid in energy_reward_info.air_handler_reward_infos:
      electrical_energy_rate += energy_reward_info.air_handler_reward_infos[
          ahid
      ].blower_electrical_energy_rate + np.abs(
          energy_reward_info.air_handler_reward_infos[
              ahid
          ].air_conditioning_electrical_energy_rate
      )

    for bid in energy_reward_info.boiler_reward_infos:
      electrical_energy_rate += energy_reward_info.boiler_reward_infos[
          bid
      ].pump_electrical_energy_rate
    return electrical_energy_rate

  def _sum_natural_gas_energy_rate(
      self, energy_reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Returns the sum of nat gas energy rate over the interval in W."""

    # Sum up the power in Watts for the total power.
    gas_energy_rate = 0.0
    for bid in energy_reward_info.boiler_reward_infos:
      gas_energy_rate += energy_reward_info.boiler_reward_infos[
          bid
      ].natural_gas_heating_energy_rate
    return gas_energy_rate
