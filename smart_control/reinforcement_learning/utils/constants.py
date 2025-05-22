"""Defines common constants used throughout the reinforcement learning codebase.

This module centralizes various fixed values, such as conversion factors,
default settings for simulations and plotting, economic parameters for reward
calculations, and reward shaping parameters. This helps in maintaining
consistency and makes it easier to adjust these values globally if needed.
"""

# --- Temperature Conversion ---
# KELVIN_TO_CELSIUS: The offset value used to convert temperatures from Kelvin
# to Celsius. The formula is: Celsius = Kelvin - KELVIN_TO_CELSIUS.
KELVIN_TO_CELSIUS = 273.15

# --- Timezone Configuration ---
# DEFAULT_TIME_ZONE: The default timezone string (e.g., 'US/Pacific', 'UTC')
# used for localizing timestamps in simulations, data processing, and plotting.
# This ensures consistency in how time is handled across different parts of
# the system.
DEFAULT_TIME_ZONE = 'US/Pacific'

# --- Economic and Productivity Constants ---
# PERSON_PRODUCTIVITY_HOUR: An economic constant representing the estimated
# monetary value (e.g., in USD) of one person's productivity for one hour.
# This can be used in reward functions to quantify the impact of occupant
# discomfort or disruptions caused by building control actions.
PERSON_PRODUCTIVITY_HOUR = 300.0

# --- Reward Shaping Parameters ---
# These constants can be used to adjust the scale and baseline of reward signals,
# which can sometimes help in stabilizing or accelerating RL training.

# REWARD_SHIFT: A constant value added to the raw calculated reward at each step.
# Can be used to shift the reward range (e.g., to make all rewards positive).
REWARD_SHIFT = 0.0 # Changed to float for consistency with REWARD_SCALE

# REWARD_SCALE: A constant value by which the raw calculated reward (after shifting)
# is multiplied. Can be used to scale the magnitude of rewards.
REWARD_SCALE = 1.0

# --- Occupancy Normalization ---
# DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT: A default value used in the
# normalization of occupancy counts or signals. This might represent a typical
# maximum expected occupancy or a scaling factor used to bring occupancy data
# into a specific range (e.g., [0, 1]) for model input or reward calculation.
DEFAULT_OCCUPANCY_NORMALIZATION_CONSTANT = 125.0
