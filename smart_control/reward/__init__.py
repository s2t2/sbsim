"""Reward function components for smart building control.

This package defines classes and functions related to calculating rewards
and costs for the reinforcement learning agent in the smart building
environment. It includes implementations for:

- Energy cost models (electricity, natural gas).
- Reward functions that combine various factors like energy consumption,
  carbon emissions, and occupant comfort/productivity (regret).

These components are used by the RL environment to provide a scalar reward
signal to the agent, guiding its learning process towards optimal building
control strategies.
"""
