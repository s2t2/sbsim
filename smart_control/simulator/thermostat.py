"""Simulates a thermostat controlling HVAC operations based on schedules.

This module defines the `Thermostat` class, which models the behavior of a
thermostat for a single zone. It determines the operational mode (e.g., HEAT,
COOL, OFF) based on the current zone temperature, a predefined setpoint
schedule, and whether the system is in "comfort" or "eco" mode.
"""

import enum
from typing import Optional # Added for type hinting

import pandas as pd

from smart_control.simulator import setpoint_schedule


class Thermostat:
  """Models a thermostat that controls heating and cooling for a zone.

  The thermostat operates based on a `SetpointSchedule`, which provides
  heating and cooling setpoints and determines if the current time falls
  within a "comfort" or "eco" period.

  In "comfort" mode, the thermostat aims to keep the zone temperature within
  the deadband defined by heating and cooling setpoints. It activates heating
  if the temperature drops below the heating setpoint and cooling if it rises
  above the cooling setpoint. It continues heating/cooling until the temperature
  reaches the midpoint of the deadband to prevent rapid cycling.

  In "eco" mode (when not in comfort mode), an additional "PASSIVE_COOL" state
  is introduced. If the system transitions from comfort to eco mode, it enters
  PASSIVE_COOL, allowing the temperature to float downwards naturally. It will
  only switch to active heating/cooling if the temperature violates the eco
  mode's (potentially wider) setpoints.

  Attributes:
    _setpoint_schedule (setpoint_schedule.SetpointSchedule): The schedule
      defining temperature setpoints and comfort/eco periods.
    _previous_timestamp (Optional[pd.Timestamp]): The timestamp of the last
      update call. Used to detect transitions between comfort/eco modes.
    _current_mode (Thermostat.Mode): The current operational mode of the
      thermostat.
  """

  class Mode(enum.Enum):
    """Defines the operational modes of the thermostat.

    Attributes:
      OFF: Temperature is within the deadband; no active heating or cooling.
      HEAT: VAV (or other HVAC unit) is actively heating the zone.
      COOL: VAV (or other HVAC unit) is actively cooling the zone.
      PASSIVE_COOL: In eco mode, the building is allowed to cool naturally
        without active cooling, typically until a lower temperature threshold
        is met or comfort mode resumes.
    """
    OFF = 0
    HEAT = 1
    COOL = 2
    PASSIVE_COOL = 3

  def __init__(self, schedule: setpoint_schedule.SetpointSchedule):
    """Initializes the Thermostat.

    Args:
      schedule (setpoint_schedule.SetpointSchedule): The setpoint schedule
        that dictates heating/cooling setpoints and comfort/eco modes based
        on time.
    """
    self._setpoint_schedule: setpoint_schedule.SetpointSchedule = schedule
    self._previous_timestamp: Optional[pd.Timestamp] = None
    self._current_mode: Thermostat.Mode = self.Mode.OFF

  def get_setpoint_schedule(self) -> setpoint_schedule.SetpointSchedule:
    """Returns the setpoint schedule used by this thermostat.

    Returns:
      setpoint_schedule.SetpointSchedule: The active setpoint schedule.
    """
    return self._setpoint_schedule

  def _default_control(
      self,
      zone_temp_k: float,
      temperature_window: setpoint_schedule.TemperatureWindow,
  ) -> 'Thermostat.Mode':
    """Determines thermostat mode based on temperature and current mode.

    This logic applies during "comfort" mode or when "eco" mode behaves
    similarly after passive cooling is no longer active. It implements
    hysteresis by continuing heating/cooling until the midpoint of the
    deadband is reached.

    Args:
      zone_temp_k (float): Current temperature (K) of the zone.
      temperature_window (setpoint_schedule.TemperatureWindow): A tuple
        (heating_setpoint_k, cooling_setpoint_k) defining the current
        comfort deadband.

    Returns:
      Thermostat.Mode: The determined operational mode (HEAT, COOL, or OFF).
    """
    heating_setpoint_k, cooling_setpoint_k = temperature_window
    midpoint_temp_k = (heating_setpoint_k + cooling_setpoint_k) / 2.0

    if zone_temp_k < heating_setpoint_k:
      self._current_mode = self.Mode.HEAT
    elif zone_temp_k > cooling_setpoint_k:
      self._current_mode = self.Mode.COOL
    # Hysteresis: If already heating and below midpoint, continue heating.
    elif zone_temp_k < midpoint_temp_k and self._current_mode == self.Mode.HEAT:
      self._current_mode = self.Mode.HEAT
    # Hysteresis: If already cooling and above midpoint, continue cooling.
    elif zone_temp_k > midpoint_temp_k and self._current_mode == self.Mode.COOL:
      self._current_mode = self.Mode.COOL
    else: # Within deadband and no active heating/cooling needed to reach midpoint
      self._current_mode = self.Mode.OFF
    return self._current_mode

  def update(
      self, zone_temp_k: float, current_timestamp: pd.Timestamp
  ) -> 'Thermostat.Mode':
    """Updates the thermostat's mode based on current conditions.

    This method should be called at each simulation step after zone
    temperatures have been updated. It determines the appropriate mode (HEAT,
    COOL, OFF, PASSIVE_COOL) by considering the zone temperature, the
    setpoint schedule, and transitions between comfort and eco modes.

    Args:
      zone_temp_k (float): The current temperature (K) of the zone.
      current_timestamp (pd.Timestamp): The current simulation timestamp.

    Returns:
      Thermostat.Mode: The new operational mode of the thermostat.
    """
    current_temp_window = self._setpoint_schedule.get_temperature_window(
        current_timestamp
    )
    is_currently_comfort_mode = self._setpoint_schedule.is_comfort_mode(
        current_timestamp
    )
    was_previously_comfort_mode = False
    if self._previous_timestamp is not None:
      was_previously_comfort_mode = self._setpoint_schedule.is_comfort_mode(
          self._previous_timestamp
      )

    if is_currently_comfort_mode:
      # Standard heating/cooling logic applies in comfort mode.
      self._default_control(zone_temp_k, current_temp_window)
    else: # Eco mode logic
      # If just transitioned from comfort to eco mode, enter passive cool.
      if self._previous_timestamp is not None and was_previously_comfort_mode:
        self._current_mode = self.Mode.PASSIVE_COOL
      else: # Already in eco mode
        # If in passive cool and temp is still above eco heating setpoint,
        # continue passive cooling.
        # (Assumes eco heating setpoint is temperature_window[0] for eco mode)
        if (self._current_mode == self.Mode.PASSIVE_COOL and
            zone_temp_k > current_temp_window[0]):
          self._current_mode = self.Mode.PASSIVE_COOL
        else:
          # Otherwise (e.g., temp dropped below eco heating setpoint, or was
          # never in passive cool), apply default control logic with eco setpoints.
          self._default_control(zone_temp_k, current_temp_window)

    self._previous_timestamp = current_timestamp
    return self._current_mode
