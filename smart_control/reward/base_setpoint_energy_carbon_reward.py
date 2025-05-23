"""Abstract base class for reward functions considering setpoints, energy, and carbon.

This module defines `BaseSetpointEnergyCarbonRewardFunction`, an abstract
class that serves as a foundation for reward functions that incorporate
occupant productivity (based on thermal comfort relative to setpoints),
energy costs, and carbon emissions.

Concrete implementations are expected to provide specific models for energy
cost and carbon emissions, and to define how these components are weighted
to form a scalar reward.
"""

from typing import Tuple

import gin
import numpy as np

from smart_control.models import base_reward_function
from smart_control.proto import smart_control_reward_pb2
from smart_control.utils import conversion_utils


@gin.configurable()
class BaseSetpointEnergyCarbonRewardFunction(
    base_reward_function.BaseRewardFunction
):
  """Calculates reward based on productivity, energy cost, and carbon emission.

  This base class provides the framework and common calculations for determining
  occupant productivity based on deviations from heating and cooling setpoints.
  Derived classes must implement the `compute_reward` method to integrate
  this productivity component with specific energy cost and carbon emission
  models to produce a final reward signal.

  The productivity model uses a logistic decay function to penalize temperatures
  outside the comfortable deadband defined by heating and cooling setpoints.

  Attributes:
    _max_productivity_personhour_usd (float): The maximum productivity value
      (e.g., in USD) per person per hour, achieved when the zone temperature
      is within the setpoint deadband.
    _productivity_midpoint_delta (float): The temperature difference (in
      Kelvin) from a setpoint at which productivity drops to 50% of its
      maximum. This defines the center of the logistic decay.
    _productivity_decay_stiffness (float): A parameter controlling the
      steepness of the logistic decay curve for productivity as temperature
      deviates from the comfortable range.
  """

  def __init__(
      self,
      max_productivity_personhour_usd: float,
      productivity_midpoint_delta: float,
      productivity_decay_stiffness: float,
  ):
    """Initializes the BaseSetpointEnergyCarbonRewardFunction.

    Args:
      max_productivity_personhour_usd (float): Maximum productivity value per
        person per hour (e.g., in USD) when conditions are comfortable.
      productivity_midpoint_delta (float): Temperature difference (K) from a
        setpoint where productivity is 50% of maximum. This determines the
        midpoint of the logistic decay for productivity loss.
      productivity_decay_stiffness (float): Controls the slope of the
        logistic decay curve for productivity. Higher values mean a steeper
        drop in productivity as temperature deviates.
    """
    self._max_productivity_personhour_usd = max_productivity_personhour_usd
    self._productivity_midpoint_delta = productivity_midpoint_delta
    self._productivity_decay_stiffness = productivity_decay_stiffness

  @abc.abstractmethod
  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Calculates the overall reward. Must be implemented by subclasses.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): Proto message
        containing all necessary data for reward calculation (energy use,
        zone temperatures, setpoints, occupancy, etc.).

    Returns:
      smart_control_reward_pb2.RewardResponse: Proto message containing the
      final agent reward and its disaggregated components.
    """
    raise NotImplementedError("Subclasses must implement compute_reward.")

  def _sum_zone_productivities(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> Tuple[float, float]:
    """Calculates total productivity and occupancy across all zones.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): Proto message
        containing zone-specific data like temperatures, setpoints, and
        average occupancy.

    Returns:
      Tuple[float, float]: A tuple where the first element is the cumulative
      estimated productivity (e.g., in USD) across all zones for the given
      time interval, and the second element is the total average occupancy
      across all zones.
    """
    time_interval_sec = self._get_delta_time_sec(reward_info)
    cumulative_productivity: float = 0.0
    total_occupancy: float = 0.0

    for zone_id in reward_info.zone_reward_infos:
      zone_info = reward_info.zone_reward_infos[zone_id]
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
      cooling_setpoint: float,
      zone_temp: float,
      time_interval_sec: float,
      average_occupancy: float,
  ) -> float:
    """Computes estimated productivity for a single zone over an interval.

    Productivity is assumed to be at maximum within the deadband defined by
    heating and cooling setpoints. Outside this deadband, it decreases
    following a logistic decay curve.

    Args:
      heating_setpoint (float): The heating setpoint temperature (K).
      cooling_setpoint (float): The cooling setpoint temperature (K).
      zone_temp (float): The actual average air temperature in the zone (K).
      time_interval_sec (float): Duration of the interval in seconds.
      average_occupancy (float): Average number of occupants in the zone during
        the interval.

    Returns:
      float: Estimated productivity (e.g., in USD) for the zone during the
      interval, considering the number of occupants.
    """
    # Midpoints for the logistic decay curves
    temp_low_midpoint = heating_setpoint - self._productivity_midpoint_delta
    temp_high_midpoint = cooling_setpoint + self._productivity_midpoint_delta

    if zone_temp < heating_setpoint: # Temperature is below heating setpoint
      # Productivity decays as temperature drops further below heating setpoint
      productivity_per_person_hour = self._max_productivity_personhour_usd / (
          1.0 + np.exp(-self._productivity_decay_stiffness *
                       (zone_temp - temp_low_midpoint))
      )
    elif zone_temp > cooling_setpoint: # Temperature is above cooling setpoint
      # Productivity decays as temperature rises further above cooling setpoint
      productivity_per_person_hour = self._max_productivity_personhour_usd * (
          1.0 - 1.0 / (
              1.0 + np.exp(-self._productivity_decay_stiffness *
                           (zone_temp - temp_high_midpoint))
          )
      )
    else: # Temperature is within the deadband
      productivity_per_person_hour = self._max_productivity_personhour_usd

    # Total productivity for the zone over the interval
    total_zone_productivity = (
        productivity_per_person_hour *
        average_occupancy *
        (time_interval_sec / 3600.0) # Convert interval to hours
    )
    return total_zone_productivity

  def _get_delta_time_sec(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Calculates the duration of the reward interval in seconds.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): Proto message
        containing start and end timestamps for the reward period.

    Returns:
      float: The duration of the interval in seconds.
    """
    start_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.start_timestamp
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.end_timestamp
    )
    return (end_time - start_time).total_seconds()

  def _sum_electricity_energy_rate(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Calculates the total electrical power consumption rate from RewardInfo.

    Sums electrical power rates from air handlers (blower and air conditioning)
    and boiler pumps. Air conditioning power is taken as absolute to account
    for both heating and cooling energy use if applicable.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): Proto message
        containing energy consumption data for various components.

    Returns:
      float: Total electrical power consumption rate in Watts.
    """
    electrical_power_watts: float = 0.0
    for ah_id in reward_info.air_handler_reward_infos:
      ah_info = reward_info.air_handler_reward_infos[ah_id]
      electrical_power_watts += ah_info.blower_electrical_energy_rate
      electrical_power_watts += np.abs(
          ah_info.air_conditioning_electrical_energy_rate
      )

    for boiler_id in reward_info.boiler_reward_infos:
      boiler_info = reward_info.boiler_reward_infos[boiler_id]
      electrical_power_watts += boiler_info.pump_electrical_energy_rate
    return electrical_power_watts

  def _sum_natural_gas_energy_rate(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> float:
    """Calculates the total natural gas consumption rate from RewardInfo.

    Sums natural gas heating power rates from all boilers.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): Proto message
        containing energy consumption data for boilers.

    Returns:
      float: Total natural gas power consumption rate in Watts.
    """
    gas_power_watts: float = 0.0
    for boiler_id in reward_info.boiler_reward_infos:
      boiler_info = reward_info.boiler_reward_infos[boiler_id]
      gas_power_watts += boiler_info.natural_gas_heating_energy_rate
    return gas_power_watts
