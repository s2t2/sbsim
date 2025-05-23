"""Models electricity cost and carbon emissions based on time-of-use rates.

This module defines the `ElectricityEnergyCost` class, which implements the
`BaseEnergyCost` interface. It calculates the monetary cost of electricity
consumption and the associated carbon emissions, considering different rates
for weekdays, weekends, and varying times of the day.

Default rates are based on typical commercial/industrial schedules (e.g., PG&E)
and Google's Carbon Free Reporting Dashboard for emission factors. These can be
overridden via Gin configuration.
"""

from typing import Sequence

from absl import logging
import gin
import numpy as np
import pandas as pd
import pint # type: ignore[import-untyped]

from smart_control.models import base_energy_cost
from smart_control.utils import conversion_utils

# Initialize a unit registry for handling physical units in calculations.
# This helps ensure dimensional consistency.
UNIT = pint.UnitRegistry()
UNIT.define("cents_per_kWh = cents / kWh") # cents per kilowatt-hour
UNIT.define("usd_per_Ws = USD / W / s")   # US dollars per Watt-second (Joule)
UNIT.define("kg_per_MWh = kg / MWh")      # kilograms per megawatt-hour
UNIT.define("Watt = J / s")               # Watt as Joule per second

# Default hourly carbon emission rates for electricity (kg CO2eq / MWh).
# Source: Google Carbon Free Reporting Dashboard, US-SVL-BORD1212.
# These values represent the carbon intensity of the electricity grid, which
# can vary by hour depending on the mix of generation sources.
_DEFAULT_CARBON_EMISSION_BY_HOUR_KG_PER_MWH: tuple = (
    88.19666493, 87.79190866, 87.87607686, 87.83054163, 88.00279618,
    88.19648183, 89.70663283, 93.97947901, 98.85868291, 100.7853521,
    101.3866866, 101.7795612, 102.5919168, 103.4403736, 104.1380294,
    104.7359292, 102.0714466, 97.04226176, 93.57895651, 92.46355045,
    91.72914657, 90.69209747, 89.76552213, 88.99950995,
)
CARBON_EMISSION_BY_HOUR: pint.Quantity = (
    _DEFAULT_CARBON_EMISSION_BY_HOUR_KG_PER_MWH * UNIT.kg_per_MWh
)

# Default time-of-use (TOU) electricity prices for weekdays (cents / kWh).
# Based on typical commercial/industrial rates (e.g., PG&E E-19 schedule,
# estimated from rate comparisons).
_DEFAULT_WEEKDAY_PRICE_CENTS_PER_KWH: tuple = (
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,  # Off-peak (e.g., 12 AM - 6 AM)
    18.0, 18.0, 18.0, 18.0, 18.0, 18.0,  # Partial-peak (e.g., 6 AM - 12 PM)
    20.0, 20.0, 20.0, 20.0, 20.0, 20.0,  # On-peak (e.g., 12 PM - 6 PM)
    16.0, 16.0, 16.0, 16.0, 16.0, 16.0,  # Off-peak (e.g., 6 PM - 12 AM)
)
WEEKDAY_PRICE_BY_HOUR: pint.Quantity = (
    _DEFAULT_WEEKDAY_PRICE_CENTS_PER_KWH * UNIT.cents_per_kWh
)

# Default electricity prices for weekends/holidays (cents / kWh).
# Often, weekend rates are flat or equivalent to off-peak weekday rates.
_DEFAULT_WEEKEND_PRICE_CENTS_PER_KWH: tuple = (16.0,) * 24 # Flat rate
WEEKEND_PRICE_BY_HOUR: pint.Quantity = (
    _DEFAULT_WEEKEND_PRICE_CENTS_PER_KWH * UNIT.cents_per_kWh
)


@gin.configurable()
class ElectricityEnergyCost(base_energy_cost.BaseEnergyCost):
  """Calculates electricity cost and carbon emissions with TOU rates.

  This class uses hourly time-of-use (TOU) pricing for weekdays and weekends,
  and hourly carbon emission factors to determine the financial cost and
  environmental impact of electricity consumption.

  Attributes:
    _carbon_emission_rates (np.ndarray): Hourly carbon emission rates in
      kg CO2eq per Watt-second (Joule).
    _weekday_energy_prices (pint.Quantity): Hourly electricity prices for
      weekdays in USD per Watt-second.
    _weekend_energy_prices (pint.Quantity): Hourly electricity prices for
      weekends/holidays in USD per Watt-second.
  """

  def __init__(
      self,
      weekday_energy_prices: Sequence[pint.Quantity] = WEEKDAY_PRICE_BY_HOUR,
      weekend_energy_prices: Sequence[pint.Quantity] = WEEKEND_PRICE_BY_HOUR,
      carbon_emission_rates: Sequence[pint.Quantity] = CARBON_EMISSION_BY_HOUR,
  ):
    """Initializes the ElectricityEnergyCost model.

    Args:
      weekday_energy_prices (Sequence[pint.Quantity]): A sequence of 24
        electricity prices for each hour of a weekday (e.g., in cents/kWh or
        USD/kWh).
      weekend_energy_prices (Sequence[pint.Quantity]): A sequence of 24
        electricity prices for each hour of a weekend/holiday.
      carbon_emission_rates (Sequence[pint.Quantity]): A sequence of 24
        carbon emission factors for each hour (e.g., in kg_CO2eq/MWh).

    Raises:
      ValueError: If any of the input sequences do not contain 24 entries.
    """
    if len(weekday_energy_prices) != 24:
      raise ValueError("Weekday energy prices must have 24 hourly entries.")
    if len(weekend_energy_prices) != 24:
      raise ValueError("Weekend energy prices must have 24 hourly entries.")
    if len(carbon_emission_rates) != 24:
      raise ValueError("Carbon emission rates must have 24 hourly entries.")

    # Convert and store carbon emission rates in kg/Ws (or kg/J)
    # Original is kg/MWh. 1 MWh = 1e6 Wh = 1e6 * 3600 Ws.
    # So, (kg/MWh) / (1e6 * 3600 Ws/MWh) = kg/Ws.
    self._carbon_emission_rates: np.ndarray = np.array([
        rate.to(UNIT.kg / UNIT.Ws).magnitude for rate in carbon_emission_rates
    ])

    # Convert and store energy prices in USD/Ws (or USD/J)
    # Original might be cents/kWh. 1 USD = 100 cents. 1 kWh = 1000 Wh = 1000*3600 Ws.
    # (cents/kWh) * (1 USD/100 cents) / (1000*3600 Ws/kWh) = USD/Ws.
    self._weekday_energy_prices: np.ndarray = np.array([
        price.to(UNIT.USD / UNIT.Ws).magnitude
        for price in weekday_energy_prices
    ])
    self._weekend_energy_prices: np.ndarray = np.array([
        price.to(UNIT.USD / UNIT.Ws).magnitude
        for price in weekend_energy_prices
    ])

  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the monetary cost of electricity consumed.

    Args:
      start_time (pd.Timestamp): The local start time of the consumption period.
      end_time (pd.Timestamp): The local end time of the consumption period.
      energy_rate (float): The average power consumed in Watts during the
        interval. A positive value indicates consumption. The absolute value
        is used for cost calculation.

    Returns:
      float: The calculated cost of electricity consumed, in USD.
    """
    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds == 0:
        return 0.0
    if duration_seconds > 3600.0:
      # Price is hourly; longer intervals might span multiple price periods.
      # For simplicity, this implementation uses the price at `start_time`.
      # More accurate models might integrate over prices if the interval
      # crosses hourly boundaries.
      logging.warning(
          "Cost query interval (%.2f hours) exceeds one hour; "
          "price at start_time will be used for the entire duration.",
          duration_seconds / 3600.0
      )

    hour_of_day = start_time.hour
    if conversion_utils.is_work_day(start_time):
      current_price_usd_per_ws = self._weekday_energy_prices[hour_of_day]
    else:
      current_price_usd_per_ws = self._weekend_energy_prices[hour_of_day]

    # Energy (Ws or Joules) = Power (W) * Duration (s)
    # Cost (USD) = Price (USD/Ws) * Energy (Ws)
    total_cost = (
        current_price_usd_per_ws * np.abs(energy_rate) * duration_seconds
    )
    return float(total_cost)

  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the carbon emissions from electricity consumption.

    Args:
      start_time (pd.Timestamp): The local start time of the consumption period.
      end_time (pd.Timestamp): The local end time of the consumption period.
      energy_rate (float): The average power consumed in Watts during the
        interval. The absolute value is used for emission calculation.

    Returns:
      float: The mass of carbon emissions (e.g., kg CO2eq) for the energy
      consumed.
    """
    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds == 0:
        return 0.0
    if duration_seconds > 3600.0:
      # Similar to cost, uses emission rate at `start_time` for simplicity.
      logging.warning(
          "Carbon query interval (%.2f hours) exceeds one hour; "
          "emission rate at start_time will be used for the entire duration.",
          duration_seconds / 3600.0
      )

    hour_of_day = start_time.hour
    current_emission_rate_kg_per_ws = self._carbon_emission_rates[hour_of_day]

    # Emissions (kg) = Emission Rate (kg/Ws) * Energy (Ws)
    # Energy (Ws) = Power (W) * Duration (s)
    total_carbon_kg = (
        current_emission_rate_kg_per_ws * np.abs(energy_rate) * duration_seconds
    )
    return float(total_carbon_kg)
