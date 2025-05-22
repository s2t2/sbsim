"""Defines the abstract base class for air convection simulators.

This module provides `BaseConvectionSimulator`, an interface that all specific
convection simulation algorithms must implement. Convection is a critical
mechanism for heat transfer within a building, driven by air movement, and
simulating it accurately is important for realistic building thermal dynamics.
Implementations of this base class will model how temperatures change due to
such airflows.
"""

import abc
from typing import MutableSequence, Dict # Used Dict for more specific type hint

import numpy as np


class BaseConvectionSimulator(metaclass=abc.ABCMeta):
  """Abstract interface for simulating air convection within a space.

  Concrete implementations of this class will provide specific algorithms
  (e.g., simple averaging within rooms, more complex computational fluid
  dynamics (CFD) approximations, or models based on pressure differences)
  to simulate how air movement distributes heat and affects the temperature
  profile of a building or a set of rooms.
  """

  @abc.abstractmethod
  def apply_convection(
      self,
      room_dict: Dict[str, MutableSequence[Tuple[int, int]]],
      temp: np.ndarray,
  ) -> None:
    """Applies the effect of air convection to a temperature field in place.

    This method simulates how air movement within and between rooms (as defined
    by `room_dict`) affects the temperature distribution (`temp`) over a single
    simulation time step. The `temp` array is modified directly by this method
    to reflect these changes.

    Args:
      room_dict: A dictionary where keys are room identifiers (e.g., strings
        like "living_room", "office_1") and values are mutable sequences of
        `(row, column)` integer tuples. Each tuple represents the coordinates
        of a grid cell belonging to that room within the temperature array.
        This structure allows the convection model to identify room boundaries
        and apply appropriate logic.
      temp: A NumPy array (typically 2D or 3D) representing the current
        temperature distribution across the simulated space. This array will be
        modified in place by the convection simulation.
    """
