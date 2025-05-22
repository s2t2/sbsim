"""Utilities for converting cyclically encoded time features.

This module provides functions to convert time representations that have been
encoded using sine and cosine transformations (a common technique to represent
cyclical features like hour of day or day of week for machine learning models)
back into more human-readable formats (e.g., integer hour or day).
"""

import numpy as np


def time_from_sin_cos(sin_theta: float, cos_theta: float) -> float:
  """Reconstructs an angle (in radians) from its sine and cosine components.

  This function is used to convert cyclically encoded time features (where a
  time point is represented by `sin(2 * pi * time / period)` and
  `cos(2 * pi * time / period)`) back into an angle representing the
  position in the cycle. The angle is returned in radians, typically in the
  range `[0, 2*pi)`.

  The implementation uses `np.arccos` and `np.arcsin` along with quadrant
  checks (implicit in the `and/or` logic) to determine the correct angle.
  Note: A more common and often more robust way to achieve this is using
  `np.arctan2(sin_theta, cos_theta)`, which directly computes the angle in the
  correct quadrant and handles edge cases like `cos_theta = 0`.

  Args:
    sin_theta: The sine component of the angle.
    cos_theta: The cosine component of the angle.

  Returns:
    The reconstructed angle in radians.

  Example:
    >>> import math
    >>> angle_rad = math.pi / 4  # 45 degrees
    >>> sin_val = math.sin(angle_rad)
    >>> cos_val = math.cos(angle_rad)
    >>> abs(time_from_sin_cos(sin_val, cos_val) - angle_rad) < 1e-9
    True
    >>> angle_rad_2 = math.pi * 1.5 # 270 degrees
    >>> sin_val_2 = math.sin(angle_rad_2)
    >>> cos_val_2 = math.cos(angle_rad_2)
    >>> abs(time_from_sin_cos(sin_val_2, cos_val_2) - angle_rad_2) < 1e-9
    True
  """
  # The original logic using 'and/or' for conditional expressions is a Python idiom
  # that relies on short-circuiting. It can be less readable than if/else.
  # `A and B or C` behaves like `B if A else C`.

  if sin_theta >= 0:
    # Quadrants I (cos_theta >= 0) or II (cos_theta < 0)
    # If cos_theta >= 0 (Quad I), angle is arccos(cos_theta).
    # If cos_theta < 0 (Quad II), angle is pi - arcsin(sin_theta)
    # (using arcsin(sin_theta) which is in [-pi/2, pi/2]).
    # The type ignore is due to pylint/pytype not fully resolving the short-circuit logic for types.
    return (
        cos_theta >= 0 and np.arccos(cos_theta) or np.pi - np.arcsin(sin_theta)  # type: ignore[return-value]
    )
  else:
    # Quadrants III (cos_theta < 0) or IV (cos_theta >= 0)
    # If cos_theta < 0 (Quad III), angle is pi - arcsin(sin_theta).
    # If cos_theta >= 0 (Quad IV), angle is 2*pi - arccos(cos_theta).
    return (
        cos_theta < 0  # type: ignore[operator]
        and np.pi - np.arcsin(sin_theta)  # type: ignore[return-value]
        or 2 * np.pi - np.arccos(cos_theta)
    )


def to_dow(sin_theta: float, cos_theta: float) -> int:
  """Converts sine and cosine encoded day-of-week back to an integer day.

  The day of the week is assumed to be encoded such that a full cycle (2*pi
  radians) corresponds to 7 days. The function reconstructs the angle from
  the sine and cosine components and then scales it to an integer day.
  The exact mapping (e.g., 0 for Monday or Sunday) depends on the original
  encoding convention. This function assumes the scaling maps to an integer
  from 0 up to 6.

  Args:
    sin_theta: The sine component of the encoded day-of-week angle.
    cos_theta: The cosine component of the encoded day-of-week angle.

  Returns:
    An integer representing the day of the week (typically 0-6).

  Example:
    >>> import math
    >>> # Example: Tuesday (day 1 if Monday is 0), angle is approx (1/7) * 2*pi
    >>> day_of_week = 1  # Assuming 0=Mon, 1=Tue, ... 6=Sun
    >>> period = 7
    >>> angle = (day_of_week / period) * 2 * math.pi
    >>> to_dow(math.sin(angle), math.cos(angle))
    1
    >>> # Example: Sunday (day 6 if Monday is 0)
    >>> day_of_week = 6
    >>> angle = (day_of_week / period) * 2 * math.pi
    >>> # For day 6, sin(6/7 * 2pi) is negative, cos is positive.
    >>> # time_from_sin_cos should yield approx 6/7 * 2pi
    >>> to_dow(math.sin(angle), math.cos(angle))
    6
  """
  theta = time_from_sin_cos(sin_theta, cos_theta)
  # Scale the angle (0 to 2*pi) to a day index (0 to 6.99...) and take floor.
  # The result is cast to int.
  return int(np.floor(7 * theta / (2 * np.pi)))


def to_hod(sin_theta: float, cos_theta: float) -> int:
  """Converts sine and cosine encoded hour-of-day back to an integer hour.

  The hour of the day is assumed to be encoded such that a full cycle (2*pi
  radians) corresponds to 24 hours. The function reconstructs the angle from
  the sine and cosine components and then scales it to an integer hour (0-23).
  The floor operation ensures an integer result.

  Args:
    sin_theta: The sine component of the encoded hour-of-day angle.
    cos_theta: The cosine component of the encoded hour-of-day angle.

  Returns:
    An integer representing the hour of the day (0-23).

  Example:
    >>> import math
    >>> # Example: 6 AM, angle is (6/24) * 2*pi = pi/2
    >>> hour = 6
    >>> period = 24
    >>> angle = (hour / period) * 2 * math.pi
    >>> to_hod(math.sin(angle), math.cos(angle))
    6
    >>> # Example: 6 PM (18:00), angle is (18/24) * 2*pi = 1.5*pi
    >>> hour = 18
    >>> angle = (hour / period) * 2 * math.pi
    >>> to_hod(math.sin(angle), math.cos(angle))
    18
  """
  theta = time_from_sin_cos(sin_theta, cos_theta)
  # Scale the angle (0 to 2*pi) to an hour index (0 to 23.99...) and take floor.
  # The result is cast to int.
  return int(np.floor(24 * theta / (2 * np.pi)))
