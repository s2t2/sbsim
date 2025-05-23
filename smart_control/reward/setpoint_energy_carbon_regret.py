"""Defines a reward function based on productivity, energy, and carbon regret.

This module implements `SetpointEnergyCarbonRegretFunction`, a concrete reward
function that calculates a scalar reward for an RL agent controlling a smart
building. The reward is formulated as a weighted combination of three
normalized "regret" components:

1.  **Productivity Regret**: Measures the loss in occupant productivity due to
    thermal discomfort (zone temperatures deviating from setpoints). This is
    normalized against a maximum possible productivity loss. A value of 0
    indicates no productivity loss (maximum comfort), while negative values
    indicate increasing loss.
2.  **Energy Cost Regret**: The actual energy cost (electricity and natural gas)
    incurred, normalized by a maximum possible energy cost. This component is
    typically negative in the reward calculation.
3.  **Carbon Emission Regret**: The amount of carbon emitted due to energy
    consumption, normalized by a maximum possible emission. This is also
    typically negative.

The final reward is a weighted sum of these normalized components, typically
scaled to be between -1 and 0, where 0 represents the ideal state (maximum
productivity, zero energy cost, zero emissions, though this is often
unachievable).

The function relies on external models for electricity and natural gas costs/
carbon emissions (`BaseEnergyCost` implementations) and inherits productivity
calculation logic from `BaseSetpointEnergyCarbonRewardFunction`.
"""

import gin

from smart_control.models import base_energy_cost
from smart_control.proto import smart_control_reward_pb2
from smart_control.reward import base_setpoint_energy_carbon_reward
from smart_control.utils import conversion_utils

_SECONDS_PER_HOUR: float = 3600.0


@gin.configurable()
class SetpointEnergyCarbonRegretFunction(
    base_setpoint_energy_carbon_reward.BaseSetpointEnergyCarbonRewardFunction
):
  """Calculates reward based on normalized productivity, energy, and carbon regret.

  The reward `r` is computed as:
  `r = (w_p * norm_prod_regret - w_e * norm_energy_cost - w_c * norm_carbon)
       / (w_p + w_e + w_c)`
  where:
    `norm_prod_regret` is the normalized productivity regret (typically <= 0).
    `norm_energy_cost` is the normalized energy cost (typically >= 0).
    `norm_carbon` is the normalized carbon emission (typically >= 0).
    `w_p, w_e, w_c` are weights for productivity, energy, and carbon.

  All normalized values are scaled to be approximately within [0, 1] before
  being used in the regret calculation (where productivity regret is adjusted
  to be negative).

  Attributes:
    _min_productivity_personhour_usd (float): Minimum assumed productivity
      value per person per hour, used for normalization.
    _max_electricity_rate (float): Maximum expected electricity consumption
      rate (Watts), used for normalizing electricity cost and carbon.
    _max_natural_gas_rate (float): Maximum expected natural gas consumption
      rate (Watts), used for normalizing gas cost and carbon.
    _electricity_energy_cost (base_energy_cost.BaseEnergyCost): Model for
      calculating electricity cost and carbon.
    _natural_gas_energy_cost (base_energy_cost.BaseEnergyCost): Model for
      calculating natural gas cost and carbon.
    _productivity_weight (float): Weight for the productivity regret component.
    _energy_cost_weight (float): Weight for the energy cost component.
    _carbon_emission_weight (float): Weight for the carbon emission component.
  """

  def __init__(
      self,
      max_productivity_personhour_usd: float,
      min_productivity_personhour_usd: float,
      max_electricity_rate_watts: float,
      max_natural_gas_rate_watts: float,
      productivity_midpoint_delta_kelvin: float,
      productivity_decay_stiffness: float,
      electricity_energy_cost_model: base_energy_cost.BaseEnergyCost,
      natural_gas_energy_cost_model: base_energy_cost.BaseEnergyCost,
      productivity_weight: float,
      energy_cost_weight: float,
      carbon_emission_weight: float,
  ):
    """Initializes the SetpointEnergyCarbonRegretFunction.

    Args:
      max_productivity_personhour_usd (float): Max productivity value
        (e.g., USD/person-hour) at optimal comfort.
      min_productivity_personhour_usd (float): Min productivity value
        (e.g., USD/person-hour) at extreme discomfort, for normalization.
      max_electricity_rate_watts (float): Max expected electricity consumption
        rate (Watts) for normalization.
      max_natural_gas_rate_watts (float): Max expected natural gas rate (Watts)
        for normalization.
      productivity_midpoint_delta_kelvin (float): Temperature deviation (K)
        from setpoint where productivity drops to 50%.
      productivity_decay_stiffness (float): Steepness of the productivity
        decay curve.
      electricity_energy_cost_model (base_energy_cost.BaseEnergyCost): Instance
        for electricity cost/carbon.
      natural_gas_energy_cost_model (base_energy_cost.BaseEnergyCost): Instance
        for natural gas cost/carbon.
      productivity_weight (float): Weight for productivity regret.
      energy_cost_weight (float): Weight for energy cost regret.
      carbon_emission_weight (float): Weight for carbon emission regret.
    """
    super().__init__(
        max_productivity_personhour_usd=max_productivity_personhour_usd,
        productivity_midpoint_delta=productivity_midpoint_delta_kelvin,
        productivity_decay_stiffness=productivity_decay_stiffness,
    )
    self._min_productivity_personhour_usd = min_productivity_personhour_usd
    self._max_electricity_rate = max_electricity_rate_watts
    self._max_natural_gas_rate = max_natural_gas_rate_watts
    self._electricity_energy_cost = electricity_energy_cost_model
    self._natural_gas_energy_cost = natural_gas_energy_cost_model
    self._productivity_weight = productivity_weight
    self._energy_cost_weight = energy_cost_weight
    self._carbon_emission_weight = carbon_emission_weight

    if self._max_productivity_personhour_usd <= self._min_productivity_personhour_usd:
      raise ValueError(
          "max_productivity_personhour_usd must be greater than "
          "min_productivity_personhour_usd for normalization."
      )
    if self._max_electricity_rate <= 0 or self._max_natural_gas_rate <= 0:
        raise ValueError("Max energy rates must be positive for normalization.")
    if self._productivity_weight < 0 or self._energy_cost_weight < 0 or \
       self._carbon_emission_weight < 0:
        raise ValueError("Reward component weights must be non-negative.")


  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Calculates the regret-based reward for the current building state.

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
    delta_time_sec = (end_time - start_time).total_seconds()
    if delta_time_sec <= 0:
        # Handle cases with no time duration, though typically unexpected.
        return smart_control_reward_pb2.RewardResponse(agent_reward_value=0.0)

    # 1. Calculate Productivity Regret
    actual_productivity_usd, total_occupancy = self._sum_zone_productivities(
        reward_info
    )
    max_possible_productivity_usd = (
        self._max_productivity_personhour_usd * total_occupancy *
        (delta_time_sec / _SECONDS_PER_HOUR)
    )
    min_possible_productivity_usd = (
        self._min_productivity_personhour_usd * total_occupancy *
        (delta_time_sec / _SECONDS_PER_HOUR)
    )
    # Ensure actual productivity is within defined min/max for stable normalization
    actual_productivity_usd = np.clip(
        actual_productivity_usd,
        min_possible_productivity_usd,
        max_possible_productivity_usd
    )

    normalized_productivity_regret: float
    if total_occupancy > 0.0 and (max_possible_productivity_usd > min_possible_productivity_usd):
      # Regret is (actual - max_achieved_under_perfect_comfort) / range
      # This should be <= 0. (actual_prod - max_prod) / (max_prod - min_prod)
      # The original formula: (actual - min) / (max - min) - 1.0 also yields a value in [-1, 0]
      normalized_productivity_regret = (
          (actual_productivity_usd - min_possible_productivity_usd) /
          (max_possible_productivity_usd - min_possible_productivity_usd)
      ) - 1.0
    else: # No occupancy or no productivity range, so no productivity regret.
      normalized_productivity_regret = 0.0

    # 2. Calculate Energy Cost Regret (Normalized Energy Cost)
    # Cap actual energy rates at maximums to prevent extreme values from dominating.
    actual_elec_rate = min(
        self._sum_electricity_energy_rate(reward_info), self._max_electricity_rate
    )
    actual_gas_rate = min(
        self._sum_natural_gas_energy_rate(reward_info), self._max_natural_gas_rate
    )

    actual_elec_cost = self._electricity_energy_cost.cost(
        start_time, end_time, actual_elec_rate
    )
    max_elec_cost = self._electricity_energy_cost.cost(
        start_time, end_time, self._max_electricity_rate
    )
    actual_gas_cost = self._natural_gas_energy_cost.cost(
        start_time, end_time, actual_gas_rate
    )
    max_gas_cost = self._natural_gas_energy_cost.cost(
        start_time, end_time, self._max_natural_gas_rate
    )

    total_actual_energy_cost = actual_elec_cost + actual_gas_cost
    total_max_energy_cost = max_elec_cost + max_gas_cost
    normalized_energy_cost: float = 0.0
    if total_max_energy_cost > 0: # Avoid division by zero
        normalized_energy_cost = total_actual_energy_cost / total_max_energy_cost
    normalized_energy_cost = np.clip(normalized_energy_cost, 0.0, 1.0) # Ensure [0,1]

    # 3. Calculate Carbon Emission Regret (Normalized Carbon Emission)
    actual_elec_carbon = self._electricity_energy_cost.carbon(
        start_time, end_time, actual_elec_rate
    )
    max_elec_carbon = self._electricity_energy_cost.carbon(
        start_time, end_time, self._max_electricity_rate
    )
    actual_gas_carbon = self._natural_gas_energy_cost.carbon(
        start_time, end_time, actual_gas_rate
    )
    max_gas_carbon = self._natural_gas_energy_cost.carbon(
        start_time, end_time, self._max_natural_gas_rate
    )

    total_actual_carbon = actual_elec_carbon + actual_gas_carbon
    total_max_carbon = max_elec_carbon + max_gas_carbon
    normalized_carbon_emission: float = 0.0
    if total_max_carbon > 0: # Avoid division by zero
        normalized_carbon_emission = total_actual_carbon / total_max_carbon
    normalized_carbon_emission = np.clip(normalized_carbon_emission, 0.0, 1.0)

    # Combine into final reward
    total_weight = (
        self._productivity_weight + self._energy_cost_weight +
        self._carbon_emission_weight
    )
    if total_weight == 0: # Avoid division by zero if all weights are zero
        agent_reward_value = 0.0
    else:
        raw_reward = (
            normalized_productivity_regret * self._productivity_weight -
            normalized_energy_cost * self._energy_cost_weight -
            normalized_carbon_emission * self._carbon_emission_weight
        )
        agent_reward_value = raw_reward / total_weight

    # Populate and return the RewardResponse proto
    response = smart_control_reward_pb2.RewardResponse(
        productivity_reward=actual_productivity_usd,
        natural_gas_energy_cost=actual_gas_cost,
        electricity_energy_cost=actual_elec_cost,
        carbon_emitted=total_actual_carbon,
        productivity_weight=self._productivity_weight,
        energy_cost_weight=self._energy_cost_weight,
        carbon_emission_weight=self._carbon_emission_weight,
        person_productivity=self._max_productivity_personhour_usd, # Max potential
        total_occupancy=total_occupancy,
        reward_scale=1.0, # Default, not actively used for scaling here
        reward_shift=0.0, # Default
        productivity_regret=(actual_productivity_usd - max_possible_productivity_usd),
        normalized_productivity_regret=normalized_productivity_regret,
        normalized_energy_cost=normalized_energy_cost,
        normalized_carbon_emission=normalized_carbon_emission,
        agent_reward_value=agent_reward_value,
    )
    response.start_timestamp.CopyFrom(reward_info.start_timestamp)
    response.end_timestamp.CopyFrom(reward_info.end_timestamp)

    return response
