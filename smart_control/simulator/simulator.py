"""Core simulation engine for building thermodynamics and HVAC interactions.

This module defines the `Simulator` class, which orchestrates the simulation
of a building's thermal behavior and the response of its HVAC (Heating,
Ventilation, and Air Conditioning) system. It uses a Finite Difference Method
(FDM) to model heat transfer within the building and with the external
environment.

The simulator manages:
- The building's thermal state (temperatures of Control Volumes).
- The HVAC system components (Air Handler, Boiler, VAVs, Thermostats).
- Interaction with a weather model for ambient conditions.
- Time progression and iterative updates of temperature and HVAC states.
- Generation of data for reward calculation in an RL context.
"""

from typing import Mapping, Tuple

from absl import logging
import gin
import numpy as np
import pandas as pd

from smart_control.models import base_occupancy
from smart_control.proto import smart_control_reward_pb2
from smart_control.simulator import building as building_py
from smart_control.simulator import hvac as hvac_py
from smart_control.simulator import weather_controller as weather_controller_py
from smart_control.utils import conversion_utils

# Type alias for reward information protobuf message.
RewardInfo = smart_control_reward_pb2.RewardInfo

# Type aliases for coordinate systems.
CVCoordinates = Tuple[int, int]  # (row, column) for Control Volumes.
ZoneId = Tuple[int, int]         # (row, column) for building zones.


@gin.configurable
class Simulator:
  """Simulates the thermodynamics of a building and its HVAC system.

  This simulator employs a Finite Difference Method (FDM) to approximate
  temperature changes across a grid of Control Volumes (CVs) representing the
  building. It iteratively calculates heat transfer (conduction, convection)
  and the impact of HVAC system outputs (heating/cooling from VAVs) at each
  time step.

  The simulation progresses by:
  1. Updating HVAC device settings based on current zone temperatures and
     thermostat logic (`setup_step_sim`).
  2. Calculating heat transfer and updating CV temperatures using FDM until
     convergence or iteration limit is reached (`finite_differences_timestep`).
  3. Applying thermal energy from HVAC to zones.
  4. Updating demands on central HVAC components (Air Handler, Boiler).
  5. Advancing the simulation clock.

  Attributes:
    building (building_py.Building): The building model instance, managing
      CV properties and temperatures.
    _hvac (hvac_py.Hvac): The HVAC system model instance.
    _weather_controller (weather_controller_py.WeatherController): Provides
      external weather conditions.
    _time_step_sec (float): Duration of each simulation time step in seconds.
    _convergence_threshold (float): Temperature change threshold (K) for FDM
      convergence.
    _iteration_limit (int): Maximum FDM iterations per time step.
    _iteration_warning (int): FDM iteration count to log a warning if not converged.
    _start_timestamp (pd.Timestamp): Initial timestamp for the simulation.
    _current_timestamp (pd.Timestamp): The current time in the simulation.
  """

  def __init__(
      self,
      building: building_py.Building,
      hvac: hvac_py.Hvac,
      weather_controller: weather_controller_py.WeatherController,
      time_step_sec: float,
      convergence_threshold: float,
      iteration_limit: int,
      iteration_warning: int,
      start_timestamp: pd.Timestamp,
  ):
    """Initializes the Simulator.

    Args:
      building (building_py.Building): An instance of the building model.
      hvac (hvac_py.Hvac): An instance of the HVAC system model.
      weather_controller (weather_controller_py.WeatherController): An instance
        that provides weather data.
      time_step_sec (float): The duration of each simulation step in seconds.
      convergence_threshold (float): The minimum temperature change (in Kelvin)
        across all CVs in an FDM iteration to consider the solution converged.
      iteration_limit (int): The maximum number of iterations for the FDM
        solver within a single time step.
      iteration_warning (int): If FDM iterations exceed this number, a warning
        is logged.
      start_timestamp (pd.Timestamp): The initial timestamp for the simulation.
        Must be timezone-aware.
    """
    self.building: building_py.Building = building
    self._hvac: hvac_py.Hvac = hvac
    self._weather_controller: weather_controller_py.WeatherController = (
        weather_controller
    )
    self._time_step_sec: float = time_step_sec
    self._convergence_threshold: float = convergence_threshold
    self._iteration_limit: int = iteration_limit
    self._iteration_warning: int = iteration_warning
    self._start_timestamp: pd.Timestamp = start_timestamp
    self._current_timestamp: pd.Timestamp = self._start_timestamp
    self.reset()

  def reset(self) -> None:
    """Resets the simulator to its initial state.

    This involves resetting the building's thermal state, the HVAC system's
    state, and setting the current simulation time back to the
    `_start_timestamp`.
    """
    self.building.reset()
    self._hvac.reset()
    self._current_timestamp = self._start_timestamp
    logging.info("Simulator reset to timestamp: %s", self._current_timestamp)

  @property
  def time_step_sec(self) -> float:
    """Duration of each simulation time step in seconds."""
    return self._time_step_sec

  @property
  def hvac(self) -> hvac_py.Hvac:
    """The HVAC system model instance."""
    return self._hvac

  @property
  def current_timestamp(self) -> pd.Timestamp:
    """The current timestamp of the simulation."""
    return self._current_timestamp

  def _get_corner_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    """Estimates temperature for a corner Control Volume (CV) for the next step.

    This calculation considers conduction with two neighboring CVs and
    convection with the ambient environment on two exposed faces.

    Args:
      cv_coordinates (CVCoordinates): (row, col) of the corner CV.
      temperature_estimates (np.ndarray): Current temperature estimates (K) for
        all CVs in the building grid.
      ambient_temperature (float): External ambient temperature (K).
      convection_coefficient (float): Convective heat transfer coefficient
        (W/m^2/K) with the ambient air.

    Returns:
      float: The estimated temperature (K) of the corner CV for the next
      FDM iteration.
    """
    x, y = cv_coordinates
    delta_x_m = self.building.cv_size_cm / 100.0 # CV side length in meters
    # Material properties at the CV
    density_kg_m3 = self.building.density[x, y]
    conductivity_W_mK = self.building.conductivity[x, y]
    heat_capacity_J_kgK = self.building.heat_capacity[x, y]
    last_temp_K = self.building.temp[x, y] # Temp from previous converged step

    neighbors_coords = self.building.neighbors[x, y]
    assert len(neighbors_coords) == 2, "Corner CV must have exactly 2 neighbors."
    neighbor_temps_K = [
        temperature_estimates[nx, ny] for nx, ny in neighbors_coords
    ]

    # FDM equation terms for a corner CV (2D explicit/implicit scheme)
    # Term related to time derivative and stored heat
    term_time_storage = (
        density_kg_m3 * (delta_x_m**2) * heat_capacity_J_kgK /
        (2.0 * self._time_step_sec) # Implicit part, hence factor of 2
    )
    retained_heat_term = term_time_storage * last_temp_K
    # Conduction from neighbors
    conduction_term = conductivity_W_mK * sum(neighbor_temps_K)
    # Convection from two faces
    convection_term = (
        2.0 * convection_coefficient * delta_x_m * ambient_temperature
    )
    # Denominator combines conduction, convection, and storage effects
    denominator = (
        2.0 * conductivity_W_mK +
        2.0 * convection_coefficient * delta_x_m +
        term_time_storage
    )
    if denominator == 0: return last_temp_K # Avoid division by zero

    return (conduction_term + convection_term + retained_heat_term) / denominator

  def _get_edge_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    """Estimates temperature for an edge CV (not corner) for the next step.

    Considers conduction with three neighboring CVs and convection with the
    ambient environment on one exposed face.

    Args:
      cv_coordinates (CVCoordinates): (row, col) of the edge CV.
      temperature_estimates (np.ndarray): Current temperature estimates (K) for
        all CVs.
      ambient_temperature (float): External ambient temperature (K).
      convection_coefficient (float): Convective heat transfer coefficient
        (W/m^2/K).

    Returns:
      float: Estimated temperature (K) of the edge CV for the next iteration.
    """
    x, y = cv_coordinates
    delta_x_m = self.building.cv_size_cm / 100.0
    density_kg_m3 = self.building.density[x, y]
    conductivity_W_mK = self.building.conductivity[x, y]
    heat_capacity_J_kgK = self.building.heat_capacity[x, y]
    last_temp_K = self.building.temp[x, y]
    neighbors_coords = self.building.neighbors[x, y]
    assert len(neighbors_coords) == 3, "Edge CV must have exactly 3 neighbors."
    neighbor_temps_K = [
        temperature_estimates[nx, ny] for nx, ny in neighbors_coords
    ]

    term_time_storage = (
        density_kg_m3 * (delta_x_m**2) * heat_capacity_J_kgK /
        (2.0 * self._time_step_sec)
    )
    retained_heat_term = term_time_storage * last_temp_K

    # Adjust conduction for edge/corner neighbors (less contact area assumed by factor)
    # This logic seems complex and might need review for physical accuracy.
    # A standard FDM formulation might not use these edge_factors directly in
    # the sum of neighbor_temps but rather in how coefficients are formed.
    # For now, preserving original logic.
    edge_factors = [
        0.5 if len(self.building.neighbors[nx, ny]) < 4 else 1.0
        for nx, ny in neighbors_coords
    ]
    conduction_term = conductivity_W_mK * sum(
        f * t for f, t in zip(edge_factors, neighbor_temps_K)
    )
    convection_term = convection_coefficient * delta_x_m * ambient_temperature
    denominator = (
        2.0 * conductivity_W_mK + # Assuming this 2.0 accounts for summation style
        convection_coefficient * delta_x_m +
        term_time_storage
    )
    if denominator == 0: return last_temp_K

    return (conduction_term + convection_term + retained_heat_term) / denominator

  def _get_interior_cv_temp_estimate(
      self, cv_coordinates: CVCoordinates, temperature_estimates: np.ndarray
  ) -> float:
    """Estimates temperature for an interior CV for the next FDM iteration.

    Considers conduction with four neighboring CVs and any internal heat
    source/sink (e.g., from HVAC diffusers).

    Args:
      cv_coordinates (CVCoordinates): (row, col) of the interior CV.
      temperature_estimates (np.ndarray): Current temperature estimates (K) for
        all CVs.

    Returns:
      float: Estimated temperature (K) of the interior CV.
    """
    x, y = cv_coordinates
    delta_x_m = self.building.cv_size_cm / 100.0
    delta_t_s = self._time_step_sec
    # Depth of CV (assumed to be floor height for 2D model heat input)
    depth_m = self.building.floor_height_cm / 100.0
    density_kg_m3 = self.building.density[x, y]
    conductivity_W_mK = self.building.conductivity[x, y]
    heat_capacity_J_kgK = self.building.heat_capacity[x, y]
    last_temp_K = self.building.temp[x, y]
    heat_input_W_m3 = self.building.input_q[x, y] # Volumetric heat source
    neighbors_coords = self.building.neighbors[x, y]
    assert len(neighbors_coords) == 4, "Interior CV must have 4 neighbors."
    neighbor_temps_K = [
        temperature_estimates[nx, ny] for nx, ny in neighbors_coords
    ]

    # Thermal diffusivity (alpha) = k / (rho * C_p)
    alpha_m2_s = conductivity_W_mK / (density_kg_m3 * heat_capacity_J_kgK)
    if alpha_m2_s == 0: return last_temp_K # Avoid division by zero if no diffusivity

    # Dimensionless Fourier number based term for time discretization
    term_fourier_inv = (delta_x_m**2) / (delta_t_s * alpha_m2_s)

    denominator = 4.0 + term_fourier_inv # For 2D conduction
    if denominator == 0: return last_temp_K

    conduction_term = sum(neighbor_temps_K)
    retained_heat_term = term_fourier_inv * last_temp_K
    # Heat source term: Q_vol * Vol / (k * A_cond_eff) -> Q_vol * dx^2*dz / (k*dz*dx_per_face_sum)
    # Simplified here as input_q / (conductivity * depth_m) - check units and derivation.
    # Assuming input_q is power (W) and needs to be scaled by area/volume correctly.
    # If input_q is W/m^3, then (input_q * delta_x^2 * depth_m) / (k*depth_m) -> (input_q * delta_x^2)/k
    # The original code has `input_q / conductivity / z` which implies input_q has units W/m.
    # This might need careful review based on how input_q is defined and applied.
    # For now, preserving.
    heat_source_term = heat_input_W_m3 / (conductivity_W_mK * depth_m)


    return (conduction_term + heat_source_term + retained_heat_term) / denominator

  def _get_cv_temp_estimate(
      self,
      cv_coordinates: CVCoordinates,
      temperature_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> float:
    """Dispatches to the correct CV temperature estimation based on type.

    Args:
      cv_coordinates (CVCoordinates): (row, col) of the CV.
      temperature_estimates (np.ndarray): Current temperature estimates (K).
      ambient_temperature (float): External ambient temperature (K).
      convection_coefficient (float): Convective heat transfer coefficient
        (W/m^2/K).

    Returns:
      float: The estimated temperature (K) for the specified CV.
    """
    num_neighbors = len(self.building.neighbors[cv_coordinates[0], cv_coordinates[1]])

    if num_neighbors <= 1: # Isolated or incorrectly defined, assume ambient
      return ambient_temperature
    elif num_neighbors == 2: # Corner CV
      return self._get_corner_cv_temp_estimate(
          cv_coordinates, temperature_estimates, ambient_temperature,
          convection_coefficient
      )
    elif num_neighbors == 3: # Edge CV
      return self._get_edge_cv_temp_estimate(
          cv_coordinates, temperature_estimates, ambient_temperature,
          convection_coefficient
      )
    else: # Interior CV (assuming num_neighbors == 4)
      return self._get_interior_cv_temp_estimate(
          cv_coordinates, temperature_estimates
      )

  def update_temperature_estimates(
      self,
      current_temp_estimates: np.ndarray,
      ambient_temperature: float,
      convection_coefficient: float,
  ) -> Tuple[np.ndarray, float]:
    """Performs one iteration of FDM temperature updates for all CVs.

    Args:
      current_temp_estimates (np.ndarray): The current grid of temperature
        estimates (K). This array will be updated in place by some methods,
        though the return is a new array.
      ambient_temperature (float): External ambient temperature (K).
      convection_coefficient (float): Convective heat transfer coefficient
        (W/m^2/K).

    Returns:
      Tuple[np.ndarray, float]:
        - next_temp_estimates (np.ndarray): The new grid of temperature
          estimates (K) after one iteration.
        - max_delta_K (float): The maximum absolute temperature change (K)
          observed in any CV during this iteration.
    """
    nrows, ncols = current_temp_estimates.shape
    next_temp_estimates = np.copy(current_temp_estimates) # Work on a copy
    max_delta_K: float = 0.0

    for r in range(nrows):
      for c in range(ncols):
        # Pass the full grid of current estimates for neighbor lookups
        new_temp_estimate_K = self._get_cv_temp_estimate(
            (r, c), current_temp_estimates, ambient_temperature,
            convection_coefficient
        )
        delta_K = abs(new_temp_estimate_K - current_temp_estimates[r, c])
        max_delta_K = max(delta_K, max_delta_K)
        next_temp_estimates[r, c] = new_temp_estimate_K

    return next_temp_estimates, max_delta_K

  def finite_differences_timestep(
      self, *, ambient_temperature: float, convection_coefficient: float
  ) -> bool:
    """Solves for CV temperatures for one simulation time step using FDM.

    This method iteratively refines temperature estimates for all Control
    Volumes (CVs) in the building until the temperature changes between
    iterations fall below a convergence threshold or a maximum iteration
    limit is reached.

    The process involves:
    1. Initializing temperature estimates (e.g., from the previous time step).
    2. Repeatedly calling `update_temperature_estimates` to get new estimates.
    3. Checking for convergence based on `_convergence_threshold`.

    Args:
      ambient_temperature (float): Current external ambient temperature (K).
      convection_coefficient (float): Current convective heat transfer
        coefficient (W/m^2/K) between external CV faces and ambient air.

    Returns:
      bool: True if the FDM solution converged within `_iteration_limit`,
      False otherwise.
    """
    # Start with current building temperatures as initial estimate for this step
    current_iteration_temps = self.building.temp.copy()
    converged = False

    for i in range(self._iteration_limit):
      next_iteration_temps, max_temp_change_K = (
          self.update_temperature_estimates(
              current_iteration_temps, ambient_temperature, convection_coefficient
          )
      )
      current_iteration_temps = next_iteration_temps

      if (i + 1) == self._iteration_warning and max_temp_change_K > self._convergence_threshold :
        logging.warning(
            "FDM: Iteration %d for timestamp %s, max_delta = %.3f K. "
            "Convergence slow.",
            i + 1, self._current_timestamp, max_temp_change_K
        )
      if max_temp_change_K <= self._convergence_threshold:
        converged = True
        logging.debug(
            "FDM converged in %d iterations for timestamp %s. Max delta: %.4f K",
            i + 1, self._current_timestamp, max_temp_change_K
            )
        break
    else: # Loop completed without break (no convergence)
      logging.warning(
          "FDM: Max iterations (%d) reached for timestamp %s. "
          "Max delta = %.3f K. Solution may not have fully converged.",
          self._iteration_limit, self._current_timestamp, max_temp_change_K
      )

    self.building.temp = current_iteration_temps # Update building state
    return converged

  def _calculate_return_water_temperature(
      self, zone_supply_temps_K: Mapping[ZoneId, float]
  ) -> float:
    """Calculates the mixed return hot water temperature to the boiler.

    This is a weighted average of the supply temperatures to zones that are
    currently using reheat, weighted by their respective reheat valve settings
    (flow rates).

    Args:
      zone_supply_temps_K (Mapping[ZoneId, float]): A map from zone IDs to
        the temperature (K) of the air supplied to that zone *after* potential
        reheating by the VAV's hot water coil.

    Returns:
      float: The calculated mixed return hot water temperature (K). Returns 0.0
      if no VAVs are currently demanding reheat.
    """
    weighted_temp_sum = 0.0
    total_valve_setting = 0.0
    for zone_id, vav_unit in self._hvac.vavs.items():
      if vav_unit.reheat_valve_setting > 0: # Only consider zones with active reheat
        weighted_temp_sum += (
            vav_unit.reheat_valve_setting * zone_supply_temps_K.get(zone_id, 0.0)
        )
        total_valve_setting += vav_unit.reheat_valve_setting

    if total_valve_setting == 0:
      # No reheat demand, return water temp might be undefined or some default.
      # Returning 0.0 might be problematic if boiler expects realistic temp.
      # Consider returning a default (e.g. avg zone temp or boiler loop temp).
      # For now, matching original logic of potential division by zero if not handled.
      return 0.0
    return weighted_temp_sum / total_valve_setting

  def setup_step_sim(self) -> None:
    """Prepares HVAC components for the next simulation sub-step.

    This primarily involves updating thermostat setpoints and VAV settings
    based on the current zone temperatures and schedules. This method should
    not change the thermal state of the building itself (CV temperatures).
    """
    avg_zone_temps_K = self.building.get_zone_average_temps()

    for zone_id, current_zone_temp_K in avg_zone_temps_K.items():
      vav_unit = self._hvac.vavs.get(zone_id)
      if vav_unit:
        # Thermostat logic is handled within VAV's update_settings
        vav_unit.update_settings(current_zone_temp_K, self._current_timestamp)
      else:
        logging.warning("No VAV unit found for zone ID: %s", zone_id)

  def execute_step_sim(self) -> None:
    """Executes one full simulation step, updating thermal and HVAC states.

    This method drives the core simulation logic:
    1. Updates HVAC component outputs (e.g., VAV airflow, reheat).
    2. Calculates heat transfer within the building using FDM.
    3. Applies thermal energy from HVAC to building zones.
    4. Updates demands on central plant (Air Handler, Boiler).
    5. Advances the simulation time.

    This method assumes `setup_step_sim` has already been called for the
    current timestamp to prepare HVAC settings.
    """
    # Get current average zone temperatures (after any thermostat actions in setup)
    avg_zone_temps_K = self.building.get_zone_average_temps()

    # Determine mixed recirculation air temperature for the Air Handler
    # Using overall average CV temperature as a proxy for mixed return air.
    recirculation_temp_K = self.building.temp.mean()
    ambient_temp_K = self._weather_controller.get_current_temp(
        self._current_timestamp
    )

    # Air Handler calculates supply air temperature before VAVs
    supply_air_temp_K = self._hvac.air_handler.get_supply_air_temp(
        recirculation_temp_K, ambient_temp_K
    )

    # Get convection coefficient for FDM
    convection_coeff_W_m2K = (
        self._weather_controller.get_air_convection_coefficient(
            self._current_timestamp
        )
    )

    # Update building CV temperatures based on conduction and convection
    self.finite_differences_timestep(
        ambient_temperature=ambient_temp_K,
        convection_coefficient=convection_coeff_W_m2K,
    )

    # Reset demands on central HVAC components for this step
    self._hvac.air_handler.reset_demand()
    self._hvac.boiler.reset_demand()

    zone_post_reheat_supply_temps_K: dict[ZoneId, float] = {}

    # Process each VAV unit
    for zone_id, current_zone_temp_K in avg_zone_temps_K.items():
      vav_unit = self._hvac.vavs.get(zone_id)
      if not vav_unit:
        continue

      # VAV calculates required heat output (q_zone_W) and actual supply temp
      q_zone_W, actual_supply_temp_K = vav_unit.output(
          current_zone_temp_K, supply_air_temp_K # Temp from AHU before reheat
      )
      zone_post_reheat_supply_temps_K[zone_id] = actual_supply_temp_K

      # Accumulate demands
      if vav_unit.flow_rate_demand > 0:
        self._hvac.air_handler.add_demand(vav_unit.flow_rate_demand)
      if vav_unit.reheat_demand > 0: # Assuming reheat_demand is hot water flow
        self._hvac.boiler.add_demand(vav_unit.reheat_demand)

      # Apply VAV's thermal power to its zone in the building model
      self.building.apply_thermal_power_zone(zone_id, q_zone_W)

    # Update boiler's return water temperature sensor based on VAV outputs
    self._hvac.boiler.return_water_temperature_sensor = (
        self._calculate_return_water_temperature(zone_post_reheat_supply_temps_K)
    )

    # Advance simulation time
    self._current_timestamp += pd.Timedelta(self._time_step_sec, unit="s")

  def _get_zone_reward_info(
      self,
      occupancy_model: base_occupancy.BaseOccupancy,
      zone_coords: ZoneId,
      zone_id_str: str,
      zone_air_temp_K: float,
  ) -> RewardInfo.ZoneRewardInfo:
    """Gathers data for a single zone to be used in reward calculation.

    Args:
      occupancy_model (base_occupancy.BaseOccupancy): Model to get occupancy.
      zone_coords (ZoneId): Coordinates of the zone.
      zone_id_str (str): String identifier of the zone.
      zone_air_temp_K (float): Average air temperature (K) of the zone.

    Returns:
      RewardInfo.ZoneRewardInfo: Protobuf message with zone-specific data.
    """
    vav_unit = self._hvac.vavs[zone_coords]
    thermostat_schedule = vav_unit.thermostat.get_setpoint_schedule()
    heat_sp_K, cool_sp_K = thermostat_schedule.get_temperature_window(
        self._current_timestamp
    )
    # Max air flow rate for this VAV (its capacity)
    vav_max_flow_m3_s = vav_unit.max_air_flow_rate
    # Actual total air flow from AHU (might be less than sum of VAV maxes)
    ah_actual_flow_m3_s = self._hvac.air_handler.air_flow_rate

    avg_occupancy = occupancy_model.average_zone_occupancy(
        zone_id_str,
        self._current_timestamp,
        self._current_timestamp + pd.Timedelta(self._time_step_sec, unit="s"),
    )

    return RewardInfo.ZoneRewardInfo(
        heating_setpoint_temperature=heat_sp_K,
        cooling_setpoint_temperature=cool_sp_K,
        zone_air_temperature=zone_air_temp_K,
        air_flow_rate_setpoint=vav_max_flow_m3_s, # VAV's own max capacity
        air_flow_rate=ah_actual_flow_m3_s, # System-level actual flow
        average_occupancy=avg_occupancy,
    )

  def _get_zone_reward_infos(
      self, occupancy_model: base_occupancy.BaseOccupancy
  ) -> Mapping[str, RewardInfo.ZoneRewardInfo]:
    """Gathers reward-relevant data for all zones.

    Args:
      occupancy_model (base_occupancy.BaseOccupancy): Occupancy model.

    Returns:
      Mapping[str, RewardInfo.ZoneRewardInfo]: A map from zone string ID to
      its `ZoneRewardInfo` proto.
    """
    zone_rewards_map: dict[str, RewardInfo.ZoneRewardInfo] = {}
    avg_zone_temps_K = self.building.get_zone_average_temps()
    for zone_coords_tuple, zone_temp_K in avg_zone_temps_K.items():
      zone_id_str = conversion_utils.zone_coordinates_to_id(zone_coords_tuple)
      zone_rewards_map[zone_id_str] = self._get_zone_reward_info(
          occupancy_model, zone_coords_tuple, zone_id_str, zone_temp_K
      )
    return zone_rewards_map

  def _get_air_handler_reward_infos(
      self,
  ) -> Mapping[str, RewardInfo.AirHandlerRewardInfo]:
    """Gathers reward-relevant data for the air handler.

    Returns:
      Mapping[str, RewardInfo.AirHandlerRewardInfo]: Map from AHU ID to its
      `AirHandlerRewardInfo` proto.
    """
    ah_info_map: dict[str, RewardInfo.AirHandlerRewardInfo] = {}
    ah = self._hvac.air_handler
    ah_id_str = ah.device_id()

    blower_power_W = (
        ah.compute_intake_fan_energy_rate() +
        ah.compute_exhaust_fan_energy_rate()
    )
    # Recirculation temperature estimation
    recirc_temp_K = self.building.temp.mean()
    ambient_temp_K = self._weather_controller.get_current_temp(
        self._current_timestamp
    )
    # Thermal power for conditioning (heating/cooling) mixed air
    conditioning_power_W = ah.compute_thermal_energy_rate(
        recirc_temp_K, ambient_temp_K
    )

    ah_info_map[ah_id_str] = RewardInfo.AirHandlerRewardInfo(
        blower_electrical_energy_rate=blower_power_W,
        air_conditioning_electrical_energy_rate=conditioning_power_W,
    )
    return ah_info_map

  def _get_boiler_reward_infos(
      self,
  ) -> Mapping[str, RewardInfo.BoilerRewardInfo]:
    """Gathers reward-relevant data for the boiler.

    Returns:
      Mapping[str, RewardInfo.BoilerRewardInfo]: Map from boiler ID to its
      `BoilerRewardInfo` proto.
    """
    boiler_info_map: dict[str, RewardInfo.BoilerRewardInfo] = {}
    boiler = self._hvac.boiler
    boiler_id_str = boiler.device_id()

    # Boiler calculations
    return_water_temp_K = boiler.return_water_temperature_sensor
    # Assuming boiler uses ambient temp for efficiency calcs, which is unusual.
    # Typically, it would be losses to its immediate surroundings.
    # Preserving original logic for now.
    ambient_temp_K = self._weather_controller.get_current_temp(
        self._current_timestamp
    )
    gas_heating_power_W = boiler.compute_thermal_energy_rate(
        return_water_temp_K, ambient_temp_K # Pass ambient for loss calculation
    )
    pump_power_W = boiler.compute_pump_power()

    boiler_info_map[boiler_id_str] = RewardInfo.BoilerRewardInfo(
        natural_gas_heating_energy_rate=gas_heating_power_W,
        pump_electrical_energy_rate=pump_power_W,
    )
    return boiler_info_map

  def reward_info(
      self, occupancy_model: base_occupancy.BaseOccupancy
  ) -> RewardInfo:
    """Aggregates data from all components for reward calculation.

    Args:
      occupancy_model (base_occupancy.BaseOccupancy): The model providing
        occupancy data for zones.

    Returns:
      RewardInfo: A protobuf message containing all data necessary for the
      reward function to compute the current step's reward.
    """
    start_ts_proto = conversion_utils.pandas_to_proto_timestamp(
        self._current_timestamp
    )
    end_ts_proto = conversion_utils.pandas_to_proto_timestamp(
        self._current_timestamp + pd.Timedelta(self._time_step_sec, unit="s")
    )

    return RewardInfo(
        start_timestamp=start_ts_proto,
        end_timestamp=end_ts_proto,
        zone_reward_infos=self._get_zone_reward_infos(occupancy_model),
        air_handler_reward_infos=self._get_air_handler_reward_infos(),
        boiler_reward_infos=self._get_boiler_reward_infos(),
    )

  def step_sim(self) -> None:
    """Advances the simulation by one time step.

    This involves a sequence of operations:
    1.  `setup_step_sim()`: HVAC components (thermostats, VAVs) determine their
        desired settings based on current building state and schedules.
    2.  `execute_step_sim()`:
        a.  Central plant (Air Handler, Boiler) determines its output based on
            aggregated demand.
        b.  Building thermal state (CV temperatures) is updated using FDM,
            considering conduction, convection with ambient, and heat from HVAC.
        c.  Thermal energy from VAVs is applied to respective zones.
        d.  Actual demands on central plant are updated based on VAV operation.
        e.  Simulation time is incremented.
    """
    self.setup_step_sim()
    self.execute_step_sim()
