"""Provides a model for calculating natural gas energy cost and carbon emissions.

This module defines `NaturalGasEnergyCost`, a concrete implementation of
`BaseEnergyCost`. It calculates the monetary cost of natural gas based on
monthly price data and determines carbon emissions using a fixed carbon
intensity factor for natural gas.

The default monthly gas prices are provided as a constant, sourced from EIA data.
These prices and the carbon intensity can be configured via Gin.
"""

from typing import Sequence

from absl import logging # For logging warnings
import gin
import numpy as np
import pandas as pd

from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.utils import constants # For KWH_PER_KFT3_GAS, JOULES_PER_KWH, GAS_CO2

# GAS_PRICE_BY_MONTH_SOURCE: A tuple of 12 float values representing the price
# of natural gas in "Dollars per Thousand Cubic Feet" for each month of the year
# (January to December). Default values are sourced from EIA data for California,
# commercial consumers, for the year 2020.
# Source: https://www.eia.gov/dnav/ng/hist/n3035ca3m.htm
GAS_PRICE_BY_MONTH_SOURCE: Tuple[float, ...] = ( # Made it a Tuple for immutability
    9.02, 8.35, 7.77, 7.26, 6.69, 6.86, # Jan - Jun
    6.77, 6.76, 6.99, 7.19, 7.96, 8.98, # Jul - Dec
)


@gin.configurable()
class NaturalGasEnergyCost(BaseEnergyCost):
  """Calculates natural gas energy cost and carbon emissions.

  This class implements the `BaseEnergyCost` interface for natural gas.
  It uses a schedule of monthly gas prices and a fixed carbon intensity factor
  to determine costs and emissions. Natural gas is assumed to be used only for
  heating, so negative energy rates (implying cooling) are treated as zero
  consumption.

  The input gas prices (e.g., in $/Thousand Cubic Feet) are converted internally
  to $/Joule for calculations. Similarly, the carbon factor is converted to
  kg CO2 / Joule.
  """

  def __init__(
      self, gas_price_by_month: Sequence[float] = GAS_PRICE_BY_MONTH_SOURCE
  ):
    """Initializes the NaturalGasEnergyCost model.

    Args:
      gas_price_by_month: A sequence of 12 float values representing the
        price of natural gas for each month (January to December), typically
        in units like "Dollars per Thousand Cubic Feet". Defaults to
        `GAS_PRICE_BY_MONTH_SOURCE`.

    Raises:
      AssertionError: If `gas_price_by_month` does not contain exactly 12 values.
    """
    assert (
        len(gas_price_by_month) == 12
    ), 'Gas price per month must have exactly 12 values.'

    # Convert the month-by-month gas price from $/1000 cubic feet to $/Joule.
    # constants.KWH_PER_KFT3_GAS: kWh per 1000 cubic feet of natural gas.
    # constants.JOULES_PER_KWH: Joules per kWh.
    self._month_gas_price_dollars_per_joule = (
        np.array(gas_price_by_month)
        / constants.KWH_PER_KFT3_GAS # $/kft3 -> $/kWh (by dividing by kWh/kft3)
        / constants.JOULES_PER_KWH   # $/kWh -> $/J (by dividing by J/kWh)
    )

    # Convert the carbon intensity from kg CO2 / 1000 cubic feet to kg CO2 / Joule.
    # constants.GAS_CO2: kg CO2 per 1000 cubic feet of natural gas.
    self._carbon_rate_kg_per_joule = (
        constants.GAS_CO2
        / constants.KWH_PER_KFT3_GAS # kgCO2/kft3 -> kgCO2/kWh
        / constants.JOULES_PER_KWH   # kgCO2/kWh -> kgCO2/J
    )

  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the monetary cost of natural gas consumed for heating.

    The cost is determined by the duration of the interval, the average
    `energy_rate` (thermal power in Watts), and the monthly gas price
    applicable at the `start_time`. Natural gas is assumed to be for heating;
    if `energy_rate` is negative, it's treated as zero consumption for cost
    calculation.

    Args:
      start_time: A `pandas.Timestamp` (timezone-aware) indicating the
        beginning of the consumption interval. The month of this timestamp
        determines which monthly price is used.
      end_time: A `pandas.Timestamp` (timezone-aware) indicating the end of the
        consumption interval.
      energy_rate: The average thermal power consumption rate in Watts [W]
        during the interval. Expected to be non-negative.

    Returns:
      The calculated cost of natural gas in USD (float) for the interval.
    """
    if energy_rate < 0.0:
      logging.warning(
          'Negative natural gas energy rate %3.2f W provided. Natural gas is assumed '
          'for heating only. Setting consumption to 0 for cost calculation.',
          energy_rate
      )
      energy_rate = 0.0

    dt_seconds = (end_time - start_time).total_seconds()
    # Month is 1-indexed (January=1), array is 0-indexed.
    current_monthly_price_dollars_per_joule = self._month_gas_price_dollars_per_joule[start_time.month - 1]

    # Total energy consumed in Joules = Power (J/s) * Duration (s)
    energy_consumed_joules = energy_rate * dt_seconds

    # Cost ($) = Price ($/J) * Energy (J)
    return float(current_monthly_price_dollars_per_joule * energy_consumed_joules)

  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the mass of carbon (kg) emitted from natural gas consumption.

    Carbon emissions are determined by the total energy consumed (derived from
    `energy_rate` and duration) and a fixed carbon intensity factor for
    natural gas (`_carbon_rate_kg_per_joule`). Natural gas is assumed for heating;
    if `energy_rate` is negative, it's treated as zero consumption.

    Args:
      start_time: A `pandas.Timestamp` (timezone-aware) indicating the
        beginning of the consumption interval.
      end_time: A `pandas.Timestamp` (timezone-aware) indicating the end of the
        consumption interval.
      energy_rate: The average thermal power consumption rate in Watts [W]
        during the interval. Expected to be non-negative.

    Returns:
      The calculated mass of carbon emissions in kilograms (kg) (float) for
      the interval.
    """
    if energy_rate < 0.0:
      logging.warning(
          'Negative natural gas energy rate %3.2f W provided. Natural gas is assumed '
          'for heating only. Setting consumption to 0 for carbon calculation.',
          energy_rate
      )
      energy_rate = 0.0

    dt_seconds = (end_time - start_time).total_seconds()
    # Total energy consumed in Joules = Power (J/s) * Duration (s)
    energy_consumed_joules = energy_rate * dt_seconds

    # Carbon (kg) = Rate (kg/J) * Energy (J)
    return float(self._carbon_rate_kg_per_joule * energy_consumed_joules)
