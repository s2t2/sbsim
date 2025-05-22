"""Defines a base class for energy cost and carbon for use in reward function.

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc

import pandas as pd


class BaseEnergyCost(metaclass=abc.ABCMeta):
  """Abstract base class for calculating energy cost and carbon emissions.

  This class defines an interface for models that determine the monetary cost
  of energy consumption and the associated carbon footprint over a given time
  interval. Implementations of this class will typically encapsulate
  utility-specific rate structures (e.g., time-of-use pricing, demand charges)
  and emission factors for different energy sources (e.g., electricity grid mix,
  natural gas).

  The calculated costs and carbon values are often used as components in the
  reward function for reinforcement learning agents controlling building systems.
  """

  @abc.abstractmethod
  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Computes the monetary cost (USD) for energy consumed over an interval.

    The fundamental calculation involves determining the total energy consumed
    and then applying the relevant pricing.
    Total energy (Joules) = `energy_rate` (Watts) * duration (seconds).
    This energy value must then be converted to the utility's billing units
    (e.g., kWh for electricity, therms for natural gas) and multiplied by the
    price for that unit at the given time.

    Implementations must handle all necessary unit conversions and incorporate
    time-varying tariffs if applicable (e.g., peak vs. off-peak electricity rates).

    Args:
      start_time: The starting timestamp (local time) of the energy consumption
        interval.
      end_time: The ending timestamp (local time) of the energy consumption
        interval.
      energy_rate: The average power consumption rate in Watts [W] over the
        interval.

    Returns:
      The total monetary cost in USD for the energy consumed during the
      specified interval.
    """
    pass

  @abc.abstractmethod
  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the mass of carbon (kg) emitted due to energy consumption.

    The conversion from energy consumed to carbon emissions is dependent on the
    energy source (e.g., electricity from a grid with a specific generation
    mix, natural gas combustion).
    Total energy (Joules) = `energy_rate` (Watts) * duration (seconds).
    This energy value is then multiplied by a source-specific carbon emission
    factor (e.g., kg CO2e / kWh).

    Implementations must use appropriate emission factors for the energy type
    and potentially for the time of consumption (e.g., varying grid carbon
    intensity).

    Args:
      start_time: The starting timestamp (local time) of the energy consumption
        interval.
      end_time: The ending timestamp (local time) of the energy consumption
        interval.
      energy_rate: The average power consumption rate in Watts [W] over the
        interval.

    Returns:
      The total mass of carbon emissions in kilograms (kg) resulting from the
      energy consumed during the specified interval.
    """
    pass
