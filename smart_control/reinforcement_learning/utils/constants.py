"""Defines project-wide constants for reinforcement learning.

This module centralizes various constant values used throughout the smart
building control reinforcement learning framework. These include physical
constants, default configuration values, and parameters for economic or
reward calculations.

Attributes:
  KELVIN_TO_CELSIUS (float): Conversion factor to subtract from Kelvin to get
    degrees Celsius.
  DEFAULT_TIME_ZONE (str): The default time zone (e.g., 'US/Pacific') used
    for timestamp localization in simulations, plotting, and data processing.
  PERSON_PRODUCTIVITY_HOUR (float): An economic constant representing the
    assumed productivity value (e.g., in USD) per person per hour. This
    can be used in reward calculations related to occupant comfort and
    productivity.
  REWARD_SHIFT (float): A constant value to shift rewards by. This can be
    used to adjust the baseline reward level.
  REWARD_SCALE (float): A factor to scale rewards by. This can be used to
    normalize or adjust the magnitude of rewards.
  DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT (float): A default value used in
    normalizing occupancy-related features in the environment.
"""

# Temperature conversion
KELVIN_TO_CELSIUS: float = 273.15
"""Conversion offset from Kelvin to Celsius (value to subtract from K)."""

# Default time zone for plotting and simulations
DEFAULT_TIME_ZONE: str = "US/Pacific"
"""Default IANA time zone string used for localizing timestamps."""

# Economic constants
PERSON_PRODUCTIVITY_HOUR: float = 300.0
"""Estimated economic value of one person's productivity per hour (e.g., USD)."""

# Reward adjustments
REWARD_SHIFT: float = 0.0
"""A constant added to all reward signals, can be used to shift the baseline."""

REWARD_SCALE: float = 1.0
"""A multiplicative factor applied to all reward signals for scaling."""

DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT: float = 125.0
"""Default constant used for normalizing occupancy counts in the environment.
This value might represent a typical or maximum expected occupancy, used in
formulas like: `(current_occupancy - NORM_CONST) / (NORM_CONST + 1)`.
"""
