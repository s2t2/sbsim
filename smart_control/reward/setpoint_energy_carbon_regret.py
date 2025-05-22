"""Defines a concrete reward function based on setpoint, energy, and carbon regret.

This module provides `SetpointEnergyCarbonRegretFunction`, an implementation
of `BaseSetpointEnergyCarbonRewardFunction`. This function calculates a reward
signal for a reinforcement learning agent controlling a smart building.

The reward is formulated as a "regret," where the ideal scenario (maximum
productivity, minimum energy cost, minimum carbon emissions) would yield a
reward closer to zero. Deviations from this ideal result in negative rewards
(penalties), encouraging the agent to minimize this regret.

The overall reward is a weighted sum of three normalized components:
1.  **Productivity Regret**: Measures the loss in occupant productivity due to
    thermal discomfort (zone temperatures deviating from setpoints). This is
    normalized against a defined maximum and minimum productivity. A value of
    0 indicates maximum productivity (no regret), while -1 indicates minimum
    productivity (maximum regret from a comfort perspective).
2.  **Energy Cost Regret**: The monetary cost of electricity and natural gas
    consumed, normalized against a maximum possible energy cost for the interval.
    A value of 0 indicates no energy cost, while 1 indicates maximum cost. This
    is then negatively weighted.
3.  **Carbon Emissions Regret**: The CO2 emissions resulting from energy
    consumption, normalized against a maximum possible emission level for the
    interval. A value of 0 indicates no emissions, while 1 indicates maximum
    emissions. This is also negatively weighted.

The weights for these components (u, v, w in the conceptual formula below) are
configurable via Gin, allowing customization of the trade-offs the agent should
learn.
  Conceptual formula:
    r_i = [ u * norm_prod_regret - v * norm_energy_cost - w * norm_carbon ] / (u+v+w)
  where `norm_prod_regret` is in [-1, 0] and `norm_energy_cost` and
  `norm_carbon` are in [0, 1]. This results in `r_i` typically in [-1, 0].
"""

import gin
import numpy as np # For np.abs, np.exp, max, np.clip
import pandas as pd # For pd.Timestamp

from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.proto import smart_control_reward_pb2
from smart_control.reward.base_setpoint_energy_carbon_reward import BaseSetpointEnergyCarbonRewardFunction
from smart_control.utils import conversion_utils

_HOUR_SEC = 3600.0  # Number of seconds in an hour, used for unit conversions.


@gin.configurable()
class SetpointEnergyCarbonRegretFunction(
    BaseSetpointEnergyCarbonRewardFunction
):
  """Calculates reward based on productivity, energy cost, and carbon emission regret.

  This function computes a scalar reward signal by:
  1. Estimating occupant productivity based on thermal comfort (deviation from
     temperature setpoints) and normalizing this into a "productivity regret"
     (typically in [-1, 0], where 0 is best).
  2. Calculating the costs of electricity and natural gas consumption and
     normalizing them against maximum potential costs for the interval (result
     in [0, 1], where 0 is best).
  3. Calculating carbon emissions from energy use and normalizing them against
     maximum potential emissions for the interval (result in [0, 1], where 0 is
     best).
  4. Combining these three normalized components using configurable weights.
     The final reward is typically in the range [-1, 0], where values closer to 0
     indicate better performance (lower regret).
  """

  @gin.configurable()
  def __init__(
      self,
      max_productivity_personhour_usd: float,
      min_productivity_personhour_usd: float,
      max_electricity_rate: float,
      max_natural_gas_rate: float,
      productivity_midpoint_delta: float,
      productivity_decay_stiffness: float,
      electricity_energy_cost: BaseEnergyCost,
      natural_gas_energy_cost: BaseEnergyCost,
      productivity_weight: float,
      energy_cost_weight: float,
      carbon_emission_weight: float,
  ):
    """Initializes the SetpointEnergyCarbonRegretFunction.

    Args:
      max_productivity_personhour_usd: The maximum assumed productivity value
        (e.g., in USD) per person, per hour, under optimal thermal conditions.
        Passed to parent `BaseSetpointEnergyCarbonRewardFunction`.
      min_productivity_personhour_usd: The minimum assumed productivity value
        (e.g., in USD) per person, per hour, under poor thermal conditions.
        This is used for normalizing the productivity regret component, ensuring
        it scales appropriately (e.g., between -1 and 0).
      max_electricity_rate: The maximum expected electricity consumption rate
        (in Watts) for the building systems. Used for normalizing electricity
        cost and associated carbon emissions. This helps bound the normalized
        cost/carbon components between 0 and 1.
      max_natural_gas_rate: The maximum expected natural gas consumption rate
        (in Watts) for the building systems. Used for normalizing natural gas
        cost and associated carbon emissions.
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
      productivity_weight: The weighting factor (u) for the normalized
        productivity regret component in the final reward calculation.
      energy_cost_weight: The weighting factor (v) for the normalized energy
        cost component (penalty).
      carbon_emission_weight: The weighting factor (w) for the normalized
        carbon emission component (penalty).
    """
    super().__init__(
        max_productivity_personhour_usd=max_productivity_personhour_usd,
        productivity_midpoint_delta=productivity_midpoint_delta,
        productivity_decay_stiffness=productivity_decay_stiffness,
    )
    self._min_productivity_personhour_usd = min_productivity_personhour_usd
    self._max_electricity_rate = max_electricity_rate
    self._max_natural_gas_rate = max_natural_gas_rate
    self._electricity_energy_cost = electricity_energy_cost
    self._natural_gas_energy_cost = natural_gas_energy_cost
    self._productivity_weight = productivity_weight
    self._energy_cost_weight = energy_cost_weight
    self._carbon_emission_weight = carbon_emission_weight

    assert (
        self._max_productivity_personhour_usd >= self._min_productivity_personhour_usd
    ), "Max productivity must be greater than or equal to min productivity."

  def compute_reward(
      self, reward_info: smart_control_reward_pb2.RewardInfo
  ) -> smart_control_reward_pb2.RewardResponse:
    """Computes the regret-based reward for the current building state.

    The method calculates actual and normalized values for productivity,
    energy cost, and carbon emissions. These normalized values are then
    combined using specified weights to produce a final agent reward.
    The reward is structured as a regret, typically in the range [-1, 0],
    where 0 represents the best possible outcome (no regret).

    The calculation involves:
    1.  Determining actual occupant productivity based on thermal comfort and
        comparing it against theoretical maximum and minimum productivity for
        the interval to get a `normalized_productivity_regret` ([-1, 0]).
    2.  Calculating actual energy costs (electricity and gas) and carbon
        emissions. Energy consumption rates are capped at predefined maximums
        (`max_electricity_rate`, `max_natural_gas_rate`) for stable normalization.
    3.  Normalizing these actual costs and emissions against the costs and
        emissions that would occur at the maximum consumption rates, yielding
        `normalized_energy_cost` and `normalized_carbon_emission` (both [0, 1]).
    4.  Combining the normalized productivity regret (which is desired to be 0)
        and the normalized cost/carbon penalties (which are desired to be 0)
        using their respective weights.
    5.  The final `agent_reward_value` is the weighted sum of these components,
        divided by the sum of weights.

    Args:
      reward_info: A `smart_control_reward_pb2.RewardInfo` protobuf message
        containing detailed building state and performance data for the last
        control interval.

    Returns:
      A populated `smart_control_reward_pb2.RewardResponse` protobuf message,
      containing the `agent_reward_value` and various intermediate calculated
      and normalized metrics for analysis and logging.
    """
    start_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.start_timestamp # pytype: disable=attribute-error
    )
    end_time = conversion_utils.proto_to_pandas_timestamp(
        reward_info.end_timestamp
    )

    delta_time_sec = (end_time - start_time).total_seconds()

    actual_productivity, total_occupancy = self._sum_zone_productivities(
        reward_info
    )

    max_productivity = (
        self._max_productivity_personhour_usd
        * total_occupancy
        * delta_time_sec
        / _HOUR_SEC
    )
    min_productivity = (
        self._min_productivity_personhour_usd
        * total_occupancy
        * delta_time_sec
        / _HOUR_SEC
    )

    actual_productivity = max(actual_productivity, min_productivity)

    if total_occupancy > 0.0:
      normalized_productivity_regret = (
          actual_productivity - min_productivity
      ) / (max_productivity - min_productivity) - 1.0
    else:
      normalized_productivity_regret = 0.0

    capped_electricity_energy_rate = min(
        self._sum_electricity_energy_rate(reward_info),
        self._max_electricity_rate,
    )

    actual_electricity_energy_cost = self._electricity_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_electricity_energy_rate,
    )

    max_electricity_energy_cost = self._electricity_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_electricity_rate,
    )

    actual_electricity_carbon_emission = self._electricity_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_electricity_energy_rate,
    )

    max_electricity_carbon_emission = self._electricity_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_electricity_rate,
    )

    capped_natural_gas_energy_rate = min(
        self._sum_natural_gas_energy_rate(reward_info),
        self._max_natural_gas_rate,
    )

    actual_natural_gas_energy_cost = self._natural_gas_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_natural_gas_energy_rate,
    )

    max_natural_gas_energy_cost = self._natural_gas_energy_cost.cost(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_natural_gas_rate,
    )

    actual_natural_gas_carbon_emission = self._natural_gas_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=capped_natural_gas_energy_rate,
    )

    max_natural_gas_carbon_emission = self._natural_gas_energy_cost.carbon(
        start_time=start_time,
        end_time=end_time,
        energy_rate=self._max_natural_gas_rate,
    )

    response = smart_control_reward_pb2.RewardResponse()
    response.productivity_reward = actual_productivity
    response.natural_gas_energy_cost = actual_natural_gas_energy_cost
    response.electricity_energy_cost = actual_electricity_energy_cost

    combined_energy_cost = (
        actual_electricity_energy_cost + actual_natural_gas_energy_cost
    )
    normalized_energy_cost = combined_energy_cost / (
        max_electricity_energy_cost + max_natural_gas_energy_cost
    )

    combined_carbon_emission = (
        actual_electricity_carbon_emission + actual_natural_gas_carbon_emission
    )

    normalized_carbon_emission = combined_carbon_emission / (
        max_electricity_carbon_emission + max_natural_gas_carbon_emission
    )

    response.carbon_emitted = combined_carbon_emission

    response.productivity_weight = self._productivity_weight
    response.energy_cost_weight = self._energy_cost_weight
    response.carbon_emission_weight = self._carbon_emission_weight

    response.person_productivity = self._max_productivity_personhour_usd
    response.total_occupancy = total_occupancy

    response.reward_scale = 1.0
    response.reward_shift = 0.0
    response.productivity_regret = actual_productivity - max_productivity
    response.normalized_productivity_regret = normalized_productivity_regret
    response.normalized_energy_cost = normalized_energy_cost
    response.normalized_carbon_emission = normalized_carbon_emission
    response.start_timestamp.CopyFrom(reward_info.start_timestamp)
    response.end_timestamp.CopyFrom(reward_info.end_timestamp)

    raw_reward_value = (
        normalized_productivity_regret * self._productivity_weight
        - normalized_energy_cost * self._energy_cost_weight
        - normalized_carbon_emission * self._carbon_emission_weight
    )

    # Return a weighted average.
    response.agent_reward_value = raw_reward_value / (
        self._productivity_weight
        + self._energy_cost_weight
        + self._carbon_emission_weight
    )

    return response
