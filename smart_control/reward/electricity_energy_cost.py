"""Provides a model for calculating electricity cost and carbon emissions.

This module defines `ElectricityEnergyCost`, a concrete implementation of
`BaseEnergyCost`. It calculates the monetary cost of electricity based on
time-of-use (TOU) pricing schedules (differentiating weekdays and weekends)
and determines carbon emissions based on hourly grid carbon intensity factors.

The default pricing and emission schedules are provided as constants, likely
derived from specific utility data (e.g., PG&E) and regional grid information
(e.g., for US-SVL-BORD1212). These schedules can be overridden via Gin
configuration.
"""

from typing import Sequence

from absl import logging # For logging warnings
import gin
import numpy as np
import pandas as pd
import pint # For handling physical units

from smart_control.models.base_energy_cost import BaseEnergyCost
from smart_control.utils import conversion_utils

# UNIT: A pint UnitRegistry instance for defining and converting physical units
# used in calculations (e.g., kWh, cents, USD, kg, MWh, Watts, seconds).
UNIT = pint.UnitRegistry()
UNIT.define("cents_per_kWh = cents / kWh") # Custom unit for energy price
UNIT.define("usd_per_Ws = USD / W / s")    # Custom unit for internal price representation
UNIT.define("kg_per_MWh = kg / MWh")       # Custom unit for carbon intensity
UNIT.define("Watt = J / s")                # Standard definition of Watt

# CARBON_EMISSION_BY_HOUR: A tuple of 24 values representing the average carbon
# emission rate (in kg CO2e per MWh of electricity consumed) for each hour of
# the day. Default values are sourced from Google Carbon Free Reporting Dashboard
# for US-SVL-BORD1212. This reflects varying grid carbon intensity.
CARBON_EMISSION_BY_HOUR = (
    88.19666493, 87.79190866, 87.87607686, 87.83054163, 88.00279618,
    88.19648183, 89.70663283, 93.97947901, 98.85868291, 100.7853521,
    101.3866866, 101.7795612, 102.5919168, 103.4403736, 104.1380294,
    104.7359292, 102.0714466, 97.04226176, 93.57895651, 92.46355045,
    91.72914657, 90.69209747, 89.76552213, 88.99950995,
) * UNIT.kg_per_MWh # type: ignore

# WEEKDAY_PRICE_BY_HOUR: A tuple of 24 values representing the electricity price
# (in US cents per kWh) for each hour of a weekday. Default values are based on
# PG&E commercial/industrial Time-of-Use (TOU) tariffs.
WEEKDAY_PRICE_BY_HOUR = (
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, # Off-peak
    18.0, 18.0, 18.0, 18.0, 18.0, 18.0, # Partial-peak
    20.0, 20.0, 20.0, 20.0, 20.0, 20.0, # Peak
    20.0, # Peak
    16.0, 16.0, 16.0, 16.0, 16.0, # Off-peak
) * UNIT.cents_per_kWh # type: ignore

# WEEKEND_PRICE_BY_HOUR: A tuple of 24 values representing the electricity price
# (in US cents per kWh) for each hour of a weekend/holiday. Default values are
# typically off-peak rates.
WEEKEND_PRICE_BY_HOUR = (
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0,
) * UNIT.cents_per_kWh # type: ignore


@gin.configurable()
class ElectricityEnergyCost(BaseEnergyCost):
  """Calculates electricity cost and carbon emissions using hourly schedules.

  This class implements the `BaseEnergyCost` interface to provide cost and
  carbon emission values specifically for electricity consumption. It uses
  predefined hourly schedules for:
  - Time-of-use (TOU) electricity pricing (different for weekdays and weekends).
  - Grid carbon intensity factors (emissions per unit of electricity).

  These schedules can be defaulted or configured via Gin.
  """

  def __init__(
      self,
      weekday_energy_prices: Sequence[pint.Quantity] = WEEKDAY_PRICE_BY_HOUR,
      weekend_energy_prices: Sequence[pint.Quantity] = WEEKEND_PRICE_BY_HOUR,
      carbon_emission_rates: Sequence[pint.Quantity] = CARBON_EMISSION_BY_HOUR,
  ):
    """Initializes the ElectricityEnergyCost model.

    Args:
      weekday_energy_prices: A sequence of 24 `pint.Quantity` values
        representing the electricity price (e.g., in "cents / kWh") for each
        hour of a weekday (0-23). Defaults to `WEEKDAY_PRICE_BY_HOUR`.
      weekend_energy_prices: A sequence of 24 `pint.Quantity` values for
        weekend hourly prices. Defaults to `WEEKEND_PRICE_BY_HOUR`.
      carbon_emission_rates: A sequence of 24 `pint.Quantity` values
        representing the carbon emission factor (e.g., in "kg / MWh") for
        electricity consumed during each hour of the day. Defaults to
        `CARBON_EMISSION_BY_HOUR`.

    Raises:
      ValueError: If any of the input schedule sequences do not contain exactly
        24 entries (one for each hour).
    """
    if len(weekday_energy_prices) != 24:
      raise ValueError("Weekday energy price rates must have 24 entries, one for each hour.")
    if len(weekend_energy_prices) != 24:
      raise ValueError("Weekend energy price rates must have 24 entries, one for each hour.")
    if len(carbon_emission_rates) != 24:
      raise ValueError("Carbon emission rates must have 24 entries, one for each hour.")

    # Convert input rates to consistent internal units (USD/Ws for price, kg/Ws for carbon)
    # Price: cents/kWh -> USD/kWh -> USD/kWs -> USD/Ws
    self._weekday_energy_prices = np.array([
        price.to(UNIT.USD / (UNIT.W * UNIT.s)).magnitude for price in weekday_energy_prices
    ])
    self._weekend_energy_prices = np.array([
        price.to(UNIT.USD / (UNIT.W * UNIT.s)).magnitude for price in weekend_energy_prices
    ])
    # Carbon: kg/MWh -> kg/Wh -> kg/Ws
    self._carbon_emission_rates = np.array([
        rate.to(UNIT.kg / (UNIT.W * UNIT.s)).magnitude for rate in carbon_emission_rates
    ])

  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the monetary cost of electricity consumed over an interval.

    The cost is determined by the duration of the interval, the average
    `energy_rate` (power in Watts), and the hourly electricity price applicable
    at the `start_time`. Prices vary by hour and between weekdays/weekends.
    The absolute value of `energy_rate` is used, so cost is incurred for
    energy consumed, regardless of whether it's for heating or cooling (if
    cooling is represented as negative power).

    Args:
      start_time: A `pandas.Timestamp` (timezone-aware) indicating the
        beginning of the consumption interval. The hour of this timestamp
        determines which hourly rate is used.
      end_time: A `pandas.Timestamp` (timezone-aware) indicating the end of the
        consumption interval.
      energy_rate: The average power consumption rate in Watts [W] during the
        interval.

    Returns:
      The calculated cost of electricity in USD (float) for the interval.
    """
    dt_seconds = (end_time - start_time).total_seconds()
    if dt_seconds > 3600.0:
      logging.warning(
          "Cost query interval (%.2f s) is greater than one hour; "
          "price estimate will be based on the start_time's hourly rate only.",
          dt_seconds
      )

    hour_index = start_time.hour
    current_price_usd_per_ws = (
        self._weekday_energy_prices[hour_index]
        if conversion_utils.is_work_day(start_time)
        else self._weekend_energy_prices[hour_index]
    )
    # Cost = Price (USD/Ws) * Power (W) * Duration (s)
    return float(current_price_usd_per_ws * np.abs(energy_rate) * dt_seconds)

  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the mass of carbon emitted due to electricity consumption.

    Carbon emissions are determined by the duration of the interval, the
    average `energy_rate` (power in Watts), and the hourly carbon emission
    factor applicable at the `start_time`. The absolute value of `energy_rate`
    is used, implying emissions are tied to the magnitude of energy generation.

    Args:
      start_time: A `pandas.Timestamp` (timezone-aware) indicating the
        beginning of the consumption interval. The hour of this timestamp
        determines which hourly emission factor is used.
      end_time: A `pandas.Timestamp` (timezone-aware) indicating the end of the
        consumption interval.
      energy_rate: The average power consumption rate in Watts [W] during the
        interval.

    Returns:
      The calculated mass of carbon emissions in kilograms (kg) (float) for
      the interval.
    """
    dt_seconds = (end_time - start_time).total_seconds()
    if dt_seconds > 3600.0:
      logging.warning(
          "Carbon query interval (%.2f s) is greater than one hour; "
          "emission estimate will be based on the start_time's hourly rate only.",
          dt_seconds
      )

    hour_index = start_time.hour
    current_emission_rate_kg_per_ws = self._carbon_emission_rates[hour_index]
    # Carbon (kg) = Rate (kg/Ws) * Power (W) * Duration (s)
    return float(current_emission_rate_kg_per_ws * np.abs(energy_rate) * dt_seconds)
