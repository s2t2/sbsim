"""Defines a reward function based on productivity, energy cost, and carbon cost.

This module implements `SetpointEnergyCarbonRewardFunction`, a concrete reward
function that calculates a scalar reward for an RL agent. The reward is a
weighted combination of:

1.  **Productivity Reward**: Estimated economic value of occupant productivity,
    which is maximized when zone temperatures are within comfort setpoints and
    decays based on deviation from these setpoints.
2.  **Energy Cost Penalty**: The monetary cost of electricity and natural gas
    consumed by the building's HVAC systems. This is a negative component.
3.  **Carbon Cost Penalty**: An imputed cost associated with the carbon
    emissions from energy consumption. This is also a negative component.

The final reward can be shifted and scaled for normalization purposes, which
can be beneficial for RL agent training.
"""

import gin

from smart_control.models import base_energy_cost
from smart_control.proto import smart_control_reward_pb2
from smart_control.reward import base_setpoint_energy_carbon_reward
from smart_control.utils import conversion_utils


@gin.configurable()
class SetpointEnergyCarbonRewardFunction(
    base_setpoint_energy_carbon_reward.BaseSetpointEnergyCarbonRewardFunction
):
  """Calculates reward from productivity, energy cost, and carbon cost.

  The reward `r` is computed as:
  `r_raw = productivity_reward - w_e * total_energy_cost - w_c * carbon_cost`
  `r_final = (r_raw - shift) / scale`
  where:
    `productivity_reward` is estimated based on thermal comfort.
    `total_energy_cost` is the sum of electricity and natural gas costs.
    `carbon_cost` is `total_carbon_emissions * carbon_cost_factor`.
    `w_e, w_c` are weights for energy and carbon costs.
    `shift`, `scale` are normalization parameters.

  Attributes:
    _electricity_energy_cost (base_energy_cost.BaseEnergyCost): Model for
      calculating electricity cost and carbon.
    _natural_gas_energy_cost (base_energy_cost.BaseEnergyCost): Model for
      calculating natural gas cost and carbon.
    _energy_cost_weight (float): Weight for the total energy cost component.
    _carbon_cost_weight (float): Weight for the carbon cost component.
    _carbon_cost_factor (float): Factor to convert kg of carbon emissions to
      a monetary cost (e.g., USD per kg CO2eq).
    _reward_normalizer_shift (float): Value to subtract from the raw reward
      for normalization.
    _reward_normalizer_scale (float): Value to divide the shifted raw reward by
      for normalization.
  """

  def __init__(
      self,
      max_productivity_personhour_usd: float,
      productivity_midpoint_delta_kelvin: float,
      productivity_decay_stiffness: float,
      electricity_energy_cost_model: base_energy_cost.BaseEnergyCost,
      natural_gas_energy_cost_model: base_energy_cost.BaseEnergyCost,
      energy_cost_weight: float,
      carbon_cost_weight: float,
      carbon_cost_factor: float,
      reward_normalizer_shift: float = 0.0,
      reward_normalizer_scale: float = 1.0,
  ):
    """Initializes the SetpointEnergyCarbonRewardFunction.

    Args:
      max_productivity_personhour_usd (float): Max productivity value
        (e.g., USD/person-hour) at optimal comfort.
      productivity_midpoint_delta_kelvin (float): Temperature deviation (K)
        from setpoint where productivity drops to 50%.
      productivity_decay_stiffness (float): Steepness of the productivity
        decay curve.
      electricity_energy_cost_model (base_energy_cost.BaseEnergyCost): Instance
        for electricity cost/carbon calculations.
      natural_gas_energy_cost_model (base_energy_cost.BaseEnergyCost): Instance
        for natural gas cost/carbon calculations.
      energy_cost_weight (float): Weight applied to the total energy cost
        penalty in the reward formula.
      carbon_cost_weight (float): Weight applied to the carbon cost penalty
        in the reward formula.
      carbon_cost_factor (float): Monetary value (e.g., USD) assigned per unit
        of carbon emission (e.g., per kg CO2eq).
      reward_normalizer_shift (float): A constant subtracted from the raw
        calculated reward. Useful for shifting the reward distribution.
      reward_normalizer_scale (float): A constant by which the shifted reward
        is divided. Useful for scaling the reward to a specific range.
        Must be non-zero.
    """
    super().__init__(
        max_productivity_personhour_usd=max_productivity_personhour_usd,
        productivity_midpoint_delta=productivity_midpoint_delta_kelvin,
        productivity_decay_stiffness=productivity_decay_stiffness,
    )
    self._electricity_energy_cost = electricity_energy_cost_model
    self._natural_gas_energy_cost = natural_gas_energy_cost_model
    self._energy_cost_weight = energy_cost_weight
    self._carbon_cost_weight = carbon_cost_weight
    self._carbon_cost_factor = carbon_cost_factor
    self._reward_normalizer_shift = reward_normalizer_shift
    if reward_normalizer_scale == 0:
        raise ValueError("reward_normalizer_scale cannot be zero.")
    self._reward_normalizer_scale = reward_normalizer_scale

  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Calculates the reward based on productivity, energy, and carbon costs.

    Args:
      reward_info (smart_control_reward_pb2.RewardInfo): Proto message
        containing raw data for reward calculation.

    Returns:
      smart_control_reward_pb2.RewardResponse: Proto message with the
      calculated agent reward and its disaggregated components.
    """
    start_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.start_timestamp
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.end_timestamp
    )

    # 1. Calculate Productivity Reward
    # `_sum_zone_productivities` returns (total_productivity_usd, total_occupancy)
    productivity_reward_usd, total_occupancy = self._sum_zone_productivities(
        reward_info
    )

    # 2. Calculate Energy Costs and Carbon Emissions
    # Electricity
    elec_energy_rate_watts = self._sum_electricity_energy_rate(reward_info)
    elec_cost_usd = self._electricity_energy_cost.cost(
        start_time, end_time, elec_energy_rate_watts
    )
    elec_carbon_kg = self._electricity_energy_cost.carbon(
        start_time, end_time, elec_energy_rate_watts
    )

    # Natural Gas
    gas_energy_rate_watts = self._sum_natural_gas_energy_rate(reward_info)
    gas_cost_usd = self._natural_gas_energy_cost.cost(
        start_time, end_time, gas_energy_rate_watts
    )
    gas_carbon_kg = self._natural_gas_energy_cost.carbon(
        start_time, end_time, gas_energy_rate_watts
    )

    # 3. Combine into Raw Reward
    total_energy_cost_usd = elec_cost_usd + gas_cost_usd
    total_carbon_emissions_kg = elec_carbon_kg + gas_carbon_kg
    carbon_cost_usd = total_carbon_emissions_kg * self._carbon_cost_factor

    raw_reward_value = (
        productivity_reward_usd -
        self._energy_cost_weight * total_energy_cost_usd -
        self._carbon_cost_weight * carbon_cost_usd
    )

    # 4. Normalize Reward
    normalized_agent_reward_value = (
        raw_reward_value - self._reward_normalizer_shift
    ) / self._reward_normalizer_scale

    # Populate and return the RewardResponse proto
    response = smart_control_reward_pb2.RewardResponse(
        productivity_reward=productivity_reward_usd,
        natural_gas_energy_cost=gas_cost_usd,
        electricity_energy_cost=elec_cost_usd,
        carbon_emitted=total_carbon_emissions_kg,
        carbon_cost=carbon_cost_usd,
        productivity_weight=self._productivity_weight, # Assuming this is for info
        energy_cost_weight=self._energy_cost_weight,   # Assuming this is for info
        carbon_emission_weight=self._carbon_cost_weight,# Assuming this is for info
        person_productivity=self._max_productivity_personhour_usd,
        total_occupancy=total_occupancy,
        reward_scale=self._reward_normalizer_scale,
        reward_shift=self._reward_normalizer_shift,
        # Normalized components are not explicitly calculated here but could be added
        agent_reward_value=normalized_agent_reward_value,
    )
    response.start_timestamp.CopyFrom(reward_info.start_timestamp)
    response.end_timestamp.CopyFrom(reward_info.end_timestamp)

    return response
