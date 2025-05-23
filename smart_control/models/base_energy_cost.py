"""Abstract base class for calculating energy cost and carbon emissions.

This module provides the `BaseEnergyCost` class, an abstract interface for
models that determine the monetary cost and carbon footprint associated with
energy consumption. Implementing classes are expected to handle specific
utility rate structures, energy sources, and emission factors.

The primary use of this class is within the reward function of a reinforcement
learning agent, allowing the agent to be penalized for high energy costs or
carbon emissions.

Copyright 2022 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc

import pandas as pd


class BaseEnergyCost(metaclass=abc.ABCMeta):
  """Abstract interface for energy cost and carbon emission calculations.

  This class defines the methods that concrete energy cost models must
  implement. These models are used to translate energy consumption (power over
  a time interval) into a monetary cost and an amount of carbon emitted.

  Implementing classes should encapsulate the logic for specific utility rate
  plans (e.g., time-of-use pricing, demand charges) and carbon intensity of
  different energy sources.

  Conceptual Example:
    A concrete implementation, `UtilityProviderX`, might look like:

    ```python
    class UtilityProviderX(BaseEnergyCost):
        def __init__(self, rate_schedule_file, emission_factors_file):
            # Load utility rates and emission factors
            self._rates = pd.read_csv(rate_schedule_file)
            self._emissions = pd.read_csv(emission_factors_file)
            # ... other initialization ...

        def cost(self, start_time, end_time, energy_rate_watts):
            # Calculate energy in kWh
            duration_hours = (end_time - start_time).total_seconds() / 3600
            energy_kwh = (energy_rate_watts / 1000) * duration_hours
            # Determine applicable rate based on start_time
            # (e.g., peak, off-peak)
            current_rate_usd_per_kwh = self._get_rate(start_time)
            return energy_kwh * current_rate_usd_per_kwh

        def carbon(self, start_time, end_time, energy_rate_watts):
            duration_hours = (end_time - start_time).total_seconds() / 3600
            energy_kwh = (energy_rate_watts / 1000) * duration_hours
            # Determine emission factor (e.g., kg_CO2/kWh)
            emission_factor = self._get_emission_factor(start_time)
            return energy_kwh * emission_factor
    ```
  """

  @abc.abstractmethod
  def cost(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Computes the monetary cost of energy consumed over an interval.

    The fundamental calculation involves determining the total energy consumed
    (energy_rate * duration) and multiplying by the applicable energy price.
    Implementations must handle conversions from Watts and seconds to the
    units used by utility providers (e.g., kWh, therms) and incorporate any
    time-dependent pricing.

    Args:
      start_time (pd.Timestamp): The local start time of the energy consumption
        period.
      end_time (pd.Timestamp): The local end time of the energy consumption
        period.
      energy_rate (float): The average power consumed in Watts over the
        interval [start_time, end_time).

    Returns:
      float: The calculated cost of energy consumed, typically in USD.
    """
    pass

  @abc.abstractmethod
  def carbon(
      self, start_time: pd.Timestamp, end_time: pd.Timestamp, energy_rate: float
  ) -> float:
    """Calculates the mass of carbon emitted due to energy consumption.

    This method estimates carbon emissions based on the energy consumed and
    the carbon intensity of the energy source(s) during the specified time
    interval. The carbon intensity can vary depending on the grid mix at
    different times.

    Args:
      start_time (pd.Timestamp): The local start time of the energy consumption
        period.
      end_time (pd.Timestamp): The local end time of the energy consumption
        period.
      energy_rate (float): The average power consumed in Watts over the
        interval [start_time, end_time).

    Returns:
      float: The mass of carbon emissions, typically in kilograms (kg) of CO2
      equivalent.
    """
    pass
