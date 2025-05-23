"""Observers for monitoring reinforcement learning agent training and evaluation.

This package provides various observer classes that can be used to track, log,
and visualize the behavior of RL agents and their interactions with the smart
building environment. Observers are typically called at different points in the
training or evaluation loop (e.g., at the end of each step or episode).

Available Observers:
  BaseObserver: An abstract base class defining the observer interface.
  CompositeObserver: An observer that groups multiple other observers, allowing
    them to be treated as a single unit.
  PrintStatusObserver: An observer that prints summary statistics and status
    updates to the console.
  RenderingObserver: An observer that can render or save visual representations
    of the environment state or agent behavior (if supported by the
    environment).
"""
