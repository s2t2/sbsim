"""Manages a collection of observers as a single observer unit.

This module defines the `CompositeObserver`, which implements the `Observer`
interface. It allows multiple individual observers to be grouped and treated
as a single entity. When the `CompositeObserver` is called or reset, it
delegates these operations to all the observers it contains.
"""

from typing import Sequence

from tf_agents.trajectories import trajectory as trajectory_lib

from smart_control.reinforcement_learning.observers.base_observer import Observer


class CompositeObserver(Observer):
  """An observer that groups and manages multiple individual observers.

  This class acts as a container for a list of other `Observer` instances.
  When its `__call__` or `reset` methods are invoked, it iterates through its
  list of contained observers and calls the corresponding method on each one.
  This pattern simplifies the management of multiple observers in a training
  or evaluation loop.

  It also provides methods to add or remove observers dynamically and a `close`
  method, although the `Observer` base class does not strictly define `close`.

  Attributes:
    _observers (list[Observer]): A list of `Observer` instances managed by this
      composite observer.

  Example:
    ```python
    # Assuming PrintStatusObserver and RenderingObserver are defined Observer types
    # from smart_control.reinforcement_learning.observers import PrintStatusObserver
    # from smart_control.reinforcement_learning.observers import RenderingObserver

    # status_logger = PrintStatusObserver()
    # renderer = RenderingObserver(render_frequency=10) # Example argument

    # Create a composite observer
    # composite = CompositeObserver([status_logger, renderer])

    # During data collection, call the composite observer
    # for trajectory in data_collection_loop:
    #     composite(trajectory) # This will call status_logger(trajectory)
    #                           # and renderer(trajectory)

    # At the end of an episode
    # composite.reset() # This will call reset on both status_logger and renderer
    ```
  """

  def __init__(self, observers: Sequence[Observer]):
    """Initializes the CompositeObserver with a sequence of observers.

    Args:
      observers (Sequence[Observer]): A sequence (e.g., list or tuple) of
        `Observer` instances to be managed by this composite observer.
    """
    self._observers: list[Observer] = list(observers)

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Processes the given trajectory with all contained observers.

    Each observer in the internal list will have its `__call__` method invoked
    with the provided `trajectory`.

    Args:
      trajectory (trajectory_lib.Trajectory): The trajectory data to be
        processed by each observer.
    """
    for observer in self._observers:
      observer(trajectory)

  def reset(self) -> None:
    """Resets the state of all contained observers.

    Each observer in the internal list will have its `reset` method invoked.
    """
    for observer in self._observers:
      observer.reset()

  def close(self) -> None:
    """Closes all contained observers that have a `close` method.

    This method attempts to call `close()` on each observer. If an observer
    does not have a `close` method, it will be skipped (AttributeError caught).
    The `Observer` base class does not define `close`, so this relies on
    individual observer implementations.
    """
    for observer in self._observers:
      try:
        # Try to call close() if the observer has this method.
        # This is not part of the base Observer interface but might be
        # implemented by specific observers for resource cleanup.
        observer.close()  # type: ignore[attr-defined]
      except AttributeError:
        # If an observer doesn't have a close method, skip it.
        pass

  def add_observer(self, observer: Observer) -> None:
    """Adds a new observer to the internal list of observers.

    Args:
      observer (Observer): The `Observer` instance to add.
    """
    self._observers.append(observer)

  def remove_observer(self, observer: Observer) -> None:
    """Removes a specific observer from the internal list.

    If the observer is not found in the list, this method does nothing.

    Args:
      observer (Observer): The `Observer` instance to remove.
    """
    if observer in self._observers:
      self._observers.remove(observer)
