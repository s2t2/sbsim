"""Provides a composite observer for managing multiple RL observers.

This module defines the `CompositeObserver` class, which allows for the
grouping and simultaneous invocation of several individual observer instances.
It acts as a container, forwarding calls like trajectory processing and reset
to each of its constituent observers.
"""

from typing import Sequence

from tf_agents.trajectories import trajectory as trajectory_lib

from smart_control.reinforcement_learning.observers.base_observer import Observer


class CompositeObserver(Observer):
  """An observer that delegates calls to a collection of other observers.

  This class provides a convenient way to manage and apply multiple observers
  (e.g., for logging, metrics collection, visualization) as if they were a
  single observer. When the `CompositeObserver` is called (e.g., with a
  trajectory or for a reset), it iterates through its list of contained
  observers and invokes the corresponding method on each one.

  Example:
    ```python
    from tf_agents.trajectories import trajectory

    # Assume MyObserver1 and MyObserver2 are concrete Observer implementations
    observer1 = MyObserver1()
    observer2 = MyObserver2()

    composite = CompositeObserver([observer1, observer2])

    # Example trajectory (replace with actual trajectory data)
    sample_trajectory = trajectory.mid(
        observation=(), action=(), policy_info=(), reward=0.0, discount=1.0
    )

    composite(sample_trajectory)  # Calls observer1(sample_trajectory) and observer2(sample_trajectory)
    composite.reset()             # Calls observer1.reset() and observer2.reset()
    ```
  """

  def __init__(self, observers: Sequence[Observer]):
    """Initializes the CompositeObserver with a list of observers.

    Args:
      observers: A sequence of `Observer` instances that will be managed by
        this composite observer.
    """
    self._observers = list(observers)

  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Processes the given trajectory with all contained observers.

    Each observer in the internal list will have its `__call__` method invoked
    with the provided `trajectory`.

    Args:
      trajectory: A `tf_agents.trajectories.trajectory.Trajectory` object
        to be processed by each observer.
    """
    for observer in self._observers:
      observer(trajectory)

  def reset(self) -> None:
    """Resets the state of all contained observers.

    This method calls the `reset()` method on each observer in its list.
    """
    for observer in self._observers:
      observer.reset()

  def close(self) -> None:
    """Closes all contained observers that have a `close` method.

    This method attempts to call `close()` on each observer in its list.
    It's important to note that the base `Observer` interface does not define
    a `close()` method. Therefore, this method relies on duck typing; only
    observers that actually implement a `close()` method will be affected.
    This is useful for observers that manage resources like file handles
    which need to be explicitly closed.
    """
    for observer in self._observers:
      # Check if the observer has a 'close' method before calling it.
      if hasattr(observer, "close") and callable(observer.close):
        observer.close()

  def add_observer(self, observer: Observer) -> None:
    """Adds an observer to the internal list.

    Args:
      observer: The `Observer` instance to add to this composite.
    """
    self._observers.append(observer)

  def remove_observer(self, observer: Observer) -> None:
    """Removes a specific observer from the internal list.

    If the specified observer is not found in the list, this method
    does nothing.

    Args:
      observer: The `Observer` instance to remove from this composite.
    """
    if observer in self._observers:
      self._observers.remove(observer)
