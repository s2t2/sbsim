"""Models the cost and carbon emissions associated with natural gas consumption.

This module defines the `NaturalGasEnergyCost` class, an implementation of the
`BaseEnergyCost` interface. It calculates the monetary cost and carbon footprint
of natural gas usage, typically for heating purposes in a building.

The model uses month-specific pricing for natural gas and a constant carbon
emission factor per unit of energy. Default pricing is based on historical data
(e.g., EIA data for California commercial consumers), and the carbon factor is
based on standard emission values for natural gas combustion. These can be
overridden via Gin configuration.
"""

from typing import Sequence

from absl import logging
import gin
import numpy as np
import pandas as pd

from smart_control.models import base_energy_cost
from smart_control.utils import constants

# Default monthly natural gas prices for commercial consumers.
# Source: U.S. Energy Information Administration (EIA), e.g., California data.
# Units: USD per Thousand Cubic Feet (Mcf).
# These values are illustrative and should be updated with relevant local data.
_DEFAULT_GAS_PRICE_USD_PER_MCF_BY_MONTH: tuple[float, ...] = (
    9.02, 8.35, 7.77, 7.26, 6.69, 6.86, # Jan - Jun
    6.77, 6.76, 6.99, 7.19, 7.96, 8.98, # Jul - Dec
)


@gin.configurable()
class NaturalGasEnergyCost(base_energy_cost.BaseEnergyCost):
  """Calculates natural gas cost and carbon emissions for reward functions.

  This class determines the cost of natural gas based on monthly pricing and
  calculates carbon emissions using a fixed rate per unit of energy.
  Natural gas is typically used for heating, so the model assumes positive
  energy rates (consumption).

  Attributes:
    _month_gas_price_usd_per_joule (np.ndarray): Monthly natural gas prices
      converted to USD per Joule.
    _carbon_rate_kg_per_joule (float): Carbon emission factor for natural gas
      in kilograms of CO2 equivalent per Joule.
  """

  def __init__(
      self,
      gas_price_by_month_usd_per_mcf: Sequence[float] = (
          _DEFAULT_GAS_PRICE_USD_PER_MCF_BY_MONTH
      )
  ):
    """Initializes the NaturalGasEnergyCost model.

    Args:
      gas_price_by_month_usd_per_mcf (Sequence[float]): A sequence of 12
        floating-point values representing the price of natural gas in USD per
        Thousand Cubic Feet (Mcf) for each month of the year (Jan to Dec).

    Raises:
      AssertionError: If `gas_price_by_month_usd_per_mcf` does not contain
        exactly 12 values.
    """
    assert len(gas_price_by_month_usd_per_mcf) == 12, (
        "Gas price per month must have exactly 12 values, one for each month."
    )

    # Convert monthly gas price from USD/Mcf to USD/Joule.
    # 1 Mcf = KWH_PER_KFT3_GAS kWh (from constants)
    # 1 kWh = JOULES_PER_KWH Joules (from constants)
    joules_per_mcf = (
        constants.KWH_PER_KFT3_GAS * constants.JOULES_PER_KWH
    )
    self._month_gas_price_usd_per_joule: np.ndarray = (
        np.array(gas_price_by_month_usd_per_mcf) / joules_per_mcf
    )

    # Convert carbon emission factor from kg CO2eq/Mcf to kg CO2eq/Joule.
    # GAS_CO2 is typically in kg CO2eq per Mcf (from constants).
    self._carbon_rate_kg_per_joule: float = constants.GAS_CO2 / joules_per_mcf

  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the monetary cost of natural gas consumed.

    Natural gas is assumed to be used only for heating, so a non-positive
    `energy_rate` will result in zero cost.

    Args:
      start_time (pd.Timestamp): The local start time of the consumption period.
      end_time (pd.Timestamp): The local end time of the consumption period.
      energy_rate (float): The average thermal power delivered by natural gas
        in Watts during the interval. Expected to be non-negative.

    Returns:
      float: The calculated cost of natural gas consumed, in USD.
    """
    if energy_rate <= 0.0:
      if energy_rate < 0.0:
        logging.warning(
            "Negative natural gas energy rate encountered: %.2f W. "
            "Cost will be calculated as 0, assuming gas is only for heating.",
            energy_rate
        )
      return 0.0 # No cost if no positive energy consumption

    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds == 0:
        return 0.0

    # Gas price is month-dependent. Month is 1-indexed (January=1).
    month_index = start_time.month - 1
    current_price_usd_per_joule = self._month_gas_price_usd_per_joule[month_index]

    # Energy (Joules) = Power (Watts) * Duration (seconds)
    energy_consumed_joules = energy_rate * duration_seconds

    # Cost (USD) = Price (USD/Joule) * Energy (Joules)
    total_cost = current_price_usd_per_joule * energy_consumed_joules
    return total_cost

  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the carbon emissions from natural gas consumption.

    Assumes natural gas is only used for heating; negative energy rates result
    in zero emissions from this source.

    Args:
      start_time (pd.Timestamp): The local start time of the consumption period.
        (Note: Currently unused as carbon rate is not time-dependent here).
      end_time (pd.Timestamp): The local end time of the consumption period.
      energy_rate (float): The average thermal power delivered by natural gas
        in Watts during the interval. Expected to be non-negative.

    Returns:
      float: The mass of carbon emissions (e.g., kg CO2eq) from natural gas
      consumption.
    """
    del start_time # Unused as carbon rate is constant per Joule for gas

    if energy_rate <= 0.0:
      if energy_rate < 0.0:
        logging.warning(
            "Negative natural gas energy rate encountered: %.2f W. "
            "Carbon emissions will be calculated as 0.",
            energy_rate
        )
      return 0.0 # No emissions if no positive energy consumption

    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds == 0:
        return 0.0

    # Energy (Joules) = Power (Watts) * Duration (seconds)
    energy_consumed_joules = energy_rate * duration_seconds

    # Emissions (kg) = Emission Rate (kg/Joule) * Energy (Joules)
    total_carbon_kg = self._carbon_rate_kg_per_joule * energy_consumed_joules
    return total_carbon_kg
