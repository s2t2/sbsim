"""Defines a concrete reward function combining productivity, energy, and carbon costs.

This module provides the `SetpointEnergyCarbonRewardFunction`, an implementation
of `BaseSetpointEnergyCarbonRewardFunction`. This function calculates a reward
signal for a reinforcement learning agent controlling a smart building.

The reward is formulated by directly combining the estimated monetary value of
occupant productivity (derived from thermal comfort) with weighted penalties
for energy consumption costs and the monetized cost of carbon emissions.
The conceptual formula is:
  `r = productivity_value - energy_cost_weight * total_energy_cost
       - carbon_cost_weight * (total_carbon_emissions * carbon_cost_factor)`

A positive reward is achieved if the productivity benefits outweigh the
weighted costs. The final reward can be further shifted and scaled.
"""

import gin
import pandas as pd # For pd.Timestamp type hint

from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.proto import smart_control_reward_pb2 # For type hints
from smart_control.reward.base_setpoint_energy_carbon_reward import BaseSetpointEnergyCarbonRewardFunction
from smart_control.utils import conversion_utils


@gin.configurable()
class SetpointEnergyCarbonRewardFunction(
    BaseSetpointEnergyCarbonRewardFunction
):
  """Calculates reward based on productivity benefits minus weighted energy and carbon costs.

  This reward function directly combines:
  1.  Estimated occupant productivity value (positive component), based on thermal
      comfort relative to setpoints.
  2.  A weighted penalty for the total monetary cost of electricity and natural
      gas consumption.
  3.  A weighted penalty for the monetized cost of carbon emissions (carbon
      emissions are first converted to a cost using `carbon_cost_factor`).

  The final reward can be shifted and scaled using `reward_normalizer_shift`
  and `reward_normalizer_scale`.
  """

  @gin.configurable()
  def __init__(
      self,
      max_productivity_personhour_usd: float,
      productivity_midpoint_delta: float,
      productivity_decay_stiffness: float,
      electricity_energy_cost: BaseEnergyCost,
      natural_gas_energy_cost: BaseEnergyCost,
      energy_cost_weight: float,
      carbon_cost_weight: float,
      carbon_cost_factor: float,
      reward_normalizer_shift: float = 0.0,
      reward_normalizer_scale: float = 1.0,
  ):
    """Initializes the SetpointEnergyCarbonRewardFunction.

    Args:
      max_productivity_personhour_usd: The maximum assumed productivity value
        (e.g., in USD) per person, per hour, under optimal thermal conditions.
        Passed to parent `BaseSetpointEnergyCarbonRewardFunction`.
      productivity_midpoint_delta: The temperature difference (e.g., in Kelvin)
        from setpoints at which productivity drops to 50%. Passed to parent.
      productivity_decay_stiffness: Parameter controlling the steepness of the
        productivity decay curve. Passed to parent.
      electricity_energy_cost: An instance of a `BaseEnergyCost` subclass
        (e.g., `ElectricityEnergyCost`) used to calculate electricity costs
        and associated carbon emissions.
      natural_gas_energy_cost: An instance of a `BaseEnergyCost` subclass
        (e.g., `NaturalGasEnergyCost`) used to calculate natural gas costs and
        associated carbon emissions.
      energy_cost_weight: The weighting factor (u) applied as a penalty to the
        total energy cost (electricity + natural gas).
      carbon_cost_weight: The weighting factor (w) applied as a penalty to the
        monetized cost of carbon emissions.
      carbon_cost_factor: A monetary value (e.g., in $/kg CO2e) used to convert
        the mass of carbon emissions into a cost, before applying the
        `carbon_cost_weight`.
      reward_normalizer_shift: A constant value to subtract from the raw
        calculated reward. This shifts the reward range. Defaults to 0.0.
      reward_normalizer_scale: A constant value by which the shifted reward is
        divided. This scales the reward range. Defaults to 1.0 (no scaling).
        Must be non-zero.
    """
    super().__init__(
        max_productivity_personhour_usd=max_productivity_personhour_usd,
        productivity_midpoint_delta=productivity_midpoint_delta,
        productivity_decay_stiffness=productivity_decay_stiffness,
    )
    self._electricity_energy_cost = electricity_energy_cost
    self._natural_gas_energy_cost = natural_gas_energy_cost
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
    """Computes the reward based on productivity, energy cost, and carbon cost.

    The calculation involves:
    1.  Calculating total productivity benefit using helper methods from the
        base class.
    2.  Calculating total electricity cost and carbon emissions using the
        provided `electricity_energy_cost` model.
    3.  Calculating total natural gas cost and carbon emissions using the
        provided `natural_gas_energy_cost` model.
    4.  Converting the combined carbon emissions to a monetary "carbon cost"
        using `_carbon_cost_factor`.
    5.  Combining these components into a raw reward:
        `raw = productivity - energy_weight * (elec_cost + gas_cost)
               - carbon_weight * carbon_cost_monetary`
    6.  Applying normalization (shift and scale) to the raw reward to get the
        final `agent_reward_value`.
    7.  Populating and returning a `RewardResponse` protobuf with these values.

    Args:
      reward_info: A `smart_control_reward_pb2.RewardInfo` protobuf message
        containing detailed building state and performance data.

    Returns:
      A populated `smart_control_reward_pb2.RewardResponse` protobuf message,
      including the final `agent_reward_value` and its constituent components.
    """
    start_time: pd.Timestamp = conversion_utils.proto_to_pandas_timestamp(
        reward_info.start_timestamp # pytype: disable=attribute-error
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.end_timestamp
    )

    productivity_reward, _ = self._sum_zone_productivities(reward_info)

    electricity_energy_rate = self._sum_electricity_energy_rate(reward_info)
    electricity_energy_cost = self._electricity_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=electricity_energy_rate,
    )
    electricity_carbon_emission = self._electricity_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=electricity_energy_rate,
    )

    natural_gas_energy_rate = self._sum_natural_gas_energy_rate(reward_info)
    natural_gas_energy_cost = self._natural_gas_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=natural_gas_energy_rate,
    )
    natural_gas_carbon_emission = self._natural_gas_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=natural_gas_energy_rate,
    )
    response = smart_control_reward_pb2.RewardResponse()
    response.productivity_reward = productivity_reward
    response.natural_gas_energy_cost = natural_gas_energy_cost
    response.electricity_energy_cost = electricity_energy_cost

    combined_carbon_emission = (
        electricity_carbon_emission + natural_gas_carbon_emission
    )
    response.carbon_emitted = combined_carbon_emission
    response.carbon_cost = combined_carbon_emission * self._carbon_cost_factor

    raw_reward_value = (
        productivity_reward
        - self._energy_cost_weight
        * (electricity_energy_cost + natural_gas_energy_cost)
        - self._carbon_cost_weight * response.carbon_cost
    )

    response.agent_reward_value = (
        raw_reward_value - self._reward_normalizer_shift
    ) / self._reward_normalizer_scale

    return response
