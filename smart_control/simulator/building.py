"""Represents the thermal model of a building composed of Control Volumes.

This module defines classes and functions for discretizing a building's
geometry into Control Volumes (CVs) and managing their thermal properties
(temperature, conductivity, heat capacity, density) and interactions. It
supports two main ways of defining building geometry:
1.  Grid-based (`Building` class): Assumes a rectangular grid of rooms, each
    with a specified shape.
2.  Floor-plan-based (`FloorPlanBasedBuilding` class): Uses a floor plan image
    and zone map to define arbitrary geometries, offering more flexibility.

Key functionalities include:
-   Assigning material properties to CVs representing air, interior walls, and
    exterior walls.
-   Calculating neighboring CVs for heat transfer calculations.
-   Distributing thermal energy from HVAC diffusers to CVs within a zone.
-   Calculating average zone temperatures and other zone-level statistics.
-   Resetting the building's thermal state.
-   (For `FloorPlanBasedBuilding`) Processing floor plan data to derive
    building structure, including exterior/interior walls and CV types.
"""

import abc
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gin
import numpy as np

from smart_control.simulator import base_convection_simulator
from smart_control.simulator import building_utils
from smart_control.simulator import constants
from smart_control.simulator import thermal_diffuser_utils

# Type aliases for clarity
Coordinates2D = Tuple[int, int]
"""Represents (row, column) coordinates in a 2D grid."""

Shape2D = Tuple[int, int]
"""Represents the (number_of_rows, number_of_columns) of a 2D grid."""

RoomIndicesDict = Dict[str, Sequence[Coordinates2D]]
"""Maps room/zone string identifiers to a sequence of CV coordinates within that room."""


@gin.configurable
@dataclasses.dataclass
class MaterialProperties:
  """Stores thermophysical properties of a material.

  These properties are essential for thermal simulations.

  Attributes:
    conductivity (float): Thermal conductivity in Watts per meter-Kelvin (W/mK).
      Measures a material's ability to conduct heat.
    heat_capacity (float): Specific heat capacity in Joules per kilogram-Kelvin
      (J/kgK). The amount of heat needed to raise the temperature of 1 kg of
      the material by 1 Kelvin.
    density (float): Density in kilograms per cubic meter (kg/m^3).
  """
  conductivity: float
  heat_capacity: float
  density: float


def _check_room_sizes(matrix_shape: Shape2D, room_shape: Shape2D) -> None:
  """Validates compatibility between overall building matrix and room shapes.

  This function is specific to the grid-based `Building` class which assumes
  a regular layout of rooms separated by single-CV-thick walls, with a
  two-CV-thick exterior wall layer.

  Args:
    matrix_shape (Shape2D): The (rows, cols) of the entire building CV matrix.
    room_shape (Shape2D): The (rows, cols) of air CVs within a single room.

  Raises:
    ValueError: If `room_shape` dimensions are not consistent with
      `matrix_shape` given the assumed wall layout.
  """
  # Total CVs = num_rooms * (CVs_per_room + 1 wall) + 1 outer_wall_start + 2 exterior_CVs
  # So, (matrix_rows - 3) must be divisible by (room_rows + 1 wall_CV)
  if (matrix_shape[0] - 3) % (room_shape[0] + 1) != 0:
    raise ValueError(
        f"Room shape rows ({room_shape[0]}) is not compatible with matrix "
        f"rows ({matrix_shape[0]}) due to wall assumptions."
    )
  if (matrix_shape[1] - 3) % (room_shape[1] + 1) != 0:
    raise ValueError(
        f"Room shape cols ({room_shape[1]}) is not compatible with matrix "
        f"cols ({matrix_shape[1]}) due to wall assumptions."
    )


def assign_building_exterior_values(
    array_to_modify: np.ndarray, value_to_assign: float
) -> None:
  """Assigns a given value to the exterior-most CVs of a building grid.

  In the grid-based `Building` model, the two outermost layers of Control
  Volumes (CVs) on all sides are typically treated as exterior walls or an
  interface with the ambient environment. This function modifies these layers.

  Args:
    array_to_modify (np.ndarray): The 2D NumPy array (e.g., representing
      conductivity, density) to be modified in-place.
    value_to_assign (float): The value to assign to these exterior CVs.
  """
  # Assign to the first two and last two columns
  array_to_modify[:, [0, 1, -2, -1]] = value_to_assign
  # Assign to the first two and last two rows
  array_to_modify[[0, 1, -2, -1], :] = value_to_assign


def assign_interior_wall_values(
    array_to_modify: np.ndarray, value_to_assign: float, room_shape: Shape2D
) -> None:
  """Assigns a given value to CVs representing interior walls in a grid layout.

  This function is specific to the grid-based `Building` model. It identifies
  CVs that act as interior walls separating rooms based on the `room_shape`
  and assigns them the `value_to_assign`. It assumes single-CV-thick walls.

  Args:
    array_to_modify (np.ndarray): The 2D NumPy array to be modified in-place.
    value_to_assign (float): The value for interior wall CVs.
    room_shape (Shape2D): The (rows, cols) of air CVs within each room.
  """
  _check_room_sizes(array_to_modify.shape, room_shape)
  nrows, ncols = array_to_modify.shape

  # Horizontal interior walls (skipping the 2 outer wall layers)
  # Iterate at intervals of (room_rows + 1 wall_CV)
  for r_idx in range(room_shape[0] + 2, nrows - 2, room_shape[0] + 1):
    array_to_modify[r_idx, 2:-2] = value_to_assign # Wall spans columns of rooms
  # Vertical interior walls
  for c_idx in range(room_shape[1] + 2, ncols - 2, room_shape[1] + 1):
    array_to_modify[2:-2, c_idx] = value_to_assign # Wall spans rows of rooms


def generate_thermal_diffusers(
    matrix_shape: Shape2D, room_shape: Shape2D
) -> np.ndarray:
  """Creates a 2D array representing thermal diffuser locations for grid rooms.

  For the grid-based `Building` model, this function places four diffusers
  symmetrically within each room. Each diffuser location is assigned a
  proportion of the total heat/coolth supplied to the zone (sums to 1 per zone).

  Args:
    matrix_shape (Shape2D): The (rows, cols) of the entire building CV matrix.
    room_shape (Shape2D): The (rows, cols) of air CVs within each room.

  Returns:
    np.ndarray: A 2D array of the same `matrix_shape` where non-zero entries
    indicate diffuser locations and their fractional contribution to heat input.
  """
  _check_room_sizes(matrix_shape, room_shape)
  num_diffusers_per_dimension = 2
  # Each diffuser contributes equally, sum of contributions per room is 1.0
  diffuser_contribution = 1.0 / (num_diffusers_per_dimension**2)

  diffusers_matrix = np.zeros(shape=matrix_shape, dtype=np.float32)
  nrows_total, ncols_total = matrix_shape
  room_rows, room_cols = room_shape

  # Calculate spacing for diffusers within a room
  # Aim for roughly 1/3 spacing from walls and between diffusers
  empty_spaces_rows = room_rows - num_diffusers_per_dimension
  step1_row = empty_spaces_rows // 3
  step2_row = room_rows - 1 - step1_row # Symmetrical from other end

  empty_spaces_cols = room_cols - num_diffusers_per_dimension
  step1_col = empty_spaces_cols // 3
  step2_col = room_cols - 1 - step1_col

  # Iterate through each room in the building grid
  # Room CVs start after the 2-CV thick outer wall layer
  for r_start_room_cv_matrix in range(2, nrows_total - 2 - room_rows, room_rows + 1):
    for c_start_room_cv_matrix in range(2, ncols_total - 2 - room_cols, room_cols + 1):
      # Place 4 diffusers in the current room
      diffusers_matrix[r_start_room_cv_matrix + step1_row, c_start_room_cv_matrix + step1_col] = diffuser_contribution
      diffusers_matrix[r_start_room_cv_matrix + step2_row, c_start_room_cv_matrix + step1_col] = diffuser_contribution
      diffusers_matrix[r_start_room_cv_matrix + step1_row, c_start_room_cv_matrix + step2_col] = diffuser_contribution
      diffusers_matrix[r_start_room_cv_matrix + step2_row, c_start_room_cv_matrix + step2_col] = diffuser_contribution
  return diffusers_matrix


def get_zone_bounds(
    zone_coordinates_in_building_grid: Coordinates2D, room_shape: Shape2D
) -> Tuple[int, int, int, int]:
  """Calculates CV index bounds for a zone in the grid-based `Building`.

  Given the (row, col) index of a room within the building's room grid,
  and the shape of air CVs in each room, this returns the min/max row/col
  indices for the air CVs of that specific zone in the overall building matrix.
  This excludes wall CVs.

  Args:
    zone_coordinates_in_building_grid (Coordinates2D): The (row, col) index of
      the zone/room within the building's grid of rooms.
    room_shape (Shape2D): The (rows, cols) of air CVs within a single room.

  Returns:
    Tuple[int, int, int, int]: A tuple (min_row, max_row, min_col, max_col)
    representing the inclusive CV indices for the air portion of the zone.
  """
  zone_row_idx, zone_col_idx = zone_coordinates_in_building_grid
  room_rows, room_cols = room_shape

  # CVs start after 2 outer wall layers. Each room adds (room_dim + 1 wall_CV).
  min_r = zone_row_idx * (room_rows + 1) + 2
  max_r = min_r + room_rows - 1
  min_c = zone_col_idx * (room_cols + 1) + 2
  max_c = min_c + room_cols - 1
  return (min_r, max_r, min_c, max_c)


def enlarge_exterior_walls(
    exterior_walls_map: building_utils.ExteriorWalls,
    interior_walls_map: building_utils.InteriorWalls,
) -> Tuple[building_utils.ExteriorWalls, building_utils.InteriorWalls]:
  """Expands exterior walls and shrinks interior walls based on floor plan.

  This function is used by `FloorPlanBasedBuilding`. It processes binary maps
  of exterior and interior walls derived from a floor plan. Exterior walls are
  "thickened" by expanding them by a defined amount, and interior walls are
  correspondingly "shrunk" where they overlap with the expanded exterior walls.
  This helps in creating a more realistic thermal barrier for exterior surfaces.

  Args:
    exterior_walls_map (building_utils.ExteriorWalls): A NumPy array where
      non-zero values indicate exterior wall CVs.
    interior_walls_map (building_utils.InteriorWalls): A NumPy array where
      non-zero values indicate interior wall CVs.

  Returns:
    Tuple[building_utils.ExteriorWalls, building_utils.InteriorWalls]:
      -   An updated `ExteriorWalls` map with thickened exterior walls.
      -   An updated `InteriorWalls` map with shrunk interior walls.
  """
  # Create binary versions (0 or 1)
  exterior_binary = np.uint8(
      exterior_walls_map == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
  )
  interior_binary = np.uint8(
      interior_walls_map == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  )

  # Enlarge exterior walls
  expanded_exterior_binary = building_utils.enlarge_component(
      exterior_binary, constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT
  )

  # Identify areas that are now part of the expanded exterior or were original walls
  all_wall_or_expanded_area = (
      expanded_exterior_binary + interior_binary + exterior_binary
  )
  # Final augmented exterior walls are where this sum is >= some threshold
  # (e.g., if it was an original wall or became part of the expansion).
  # The constant WALLS_AND_EXPANDED_BOOLS likely defines this threshold.
  final_exterior_augmented = np.int16(
      all_wall_or_expanded_area >= constants.WALLS_AND_EXPANDED_BOOLS
  ) * constants.EXTERIOR_WALL_VALUE_IN_FUNCTION

  # Shrink interior walls: if an interior wall CV is now part of the
  # augmented exterior wall, it's no longer considered an interior wall.
  final_interior_shrunk = np.int16(
      (interior_walls_map + final_exterior_augmented) ==
      constants.INTERIOR_WALL_VALUE_IN_FUNCTION # Only true if it was interior and not exterior
  ) * constants.INTERIOR_WALL_VALUE_IN_FUNCTION

  return final_exterior_augmented, final_interior_shrunk


def _assign_interior_and_exterior_values(
    exterior_walls_map: np.ndarray,
    interior_walls_map: np.ndarray,
    interior_wall_value: float,
    exterior_wall_value: float,
    air_space_value: float,
) -> np.ndarray:
  """Assigns material property values based on pre-defined wall and air maps.

  This function creates a property matrix (e.g., for conductivity) by assigning
  specific values to CVs identified as exterior walls, interior walls, or
  air spaces, based on input binary maps.

  Args:
    exterior_walls_map (np.ndarray): Binary map where non-zero indicates
      exterior wall CVs.
    interior_walls_map (np.ndarray): Binary map where non-zero indicates
      interior wall CVs.
    interior_wall_value (float): Property value for interior walls.
    exterior_wall_value (float): Property value for exterior walls.
    air_space_value (float): Property value for CVs that are neither interior
      nor exterior walls (i.e., air or other internal spaces).

  Returns:
    np.ndarray: A 2D array with assigned property values.
  """
  # Start by assuming all is air_space_value
  property_array = np.full_like(exterior_walls_map, air_space_value, dtype=float)
  # Overlay interior wall values
  property_array[interior_walls_map == constants.INTERIOR_WALL_VALUE_IN_FUNCTION] = interior_wall_value
  # Overlay exterior wall values (takes precedence over interior if overlap)
  property_array[exterior_walls_map == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION] = exterior_wall_value
  return property_array


def _construct_cv_type_array(
    exterior_walls_map: np.ndarray, exterior_space_map: np.ndarray
) -> np.ndarray:
  """Creates a matrix identifying the type of each Control Volume (CV).

  Based on binary maps for exterior walls and exterior space (ambient air),
  this function categorizes each CV as:
  -   Exterior Space (e.g., outside air)
  -   Interior Space (e.g., room air, not a wall)
  -   Wall (either interior or exterior, not explicitly distinguished here beyond
    not being space)

  Args:
    exterior_walls_map (np.ndarray): Binary map of exterior wall locations.
    exterior_space_map (np.ndarray): Binary map of exterior space locations.

  Returns:
    np.ndarray: A 2D array of strings, where each entry is a label like
    `constants.LABEL_FOR_EXTERIOR_SPACE`.
  """
  # Default to wall
  cv_type_matrix = np.full_like(
      exterior_walls_map, constants.LABEL_FOR_WALLS, dtype=object
  )
  # Interior space is where there are no exterior walls AND no exterior space
  # (This assumes exterior_walls_map marks only the structural part of walls,
  # and interior_space_value is 0 for walls in that map).
  # A common convention: floor_plan has 0 for air, 1 for interior wall, 2 for exterior.
  # If exterior_walls_map comes from such a floor_plan (e.g. value 2), then
  # `exterior_walls == INTERIOR_SPACE_VALUE (0)` would be true for air and int walls.
  # This needs careful interpretation of `constants.INTERIOR_SPACE_VALUE_IN_FUNCTION`.
  # Assuming constants.INTERIOR_SPACE_VALUE_IN_FUNCTION (e.g. 0) means NOT an exterior wall.
  cv_type_matrix[
      exterior_walls_map == constants.INTERIOR_SPACE_VALUE_IN_FUNCTION
  ] = constants.LABEL_FOR_INTERIOR_SPACE
  # Exterior space overrides all
  cv_type_matrix[
      exterior_space_map == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION
  ] = constants.LABEL_FOR_EXTERIOR_SPACE
  return cv_type_matrix


def _assign_thermal_diffusers(
    diffuser_matrix_to_fill: np.ndarray,
    room_cv_indices_map: RoomIndicesDict,
    interior_walls_map: building_utils.InteriorWalls,
    diffuser_spacing_cvs: int = 10,
    buffer_from_walls_cvs: int = 5,
) -> np.ndarray:
  """Distributes thermal diffuser locations within zones based on geometry.

  This function places thermal diffusers (points of heat/coolth injection)
  within each defined room/zone. It attempts to space them based on
  `diffuser_spacing_cvs` and maintain a `buffer_from_walls_cvs`. For non-
  rectangular rooms, it may use a random allocation strategy. The values in
  the output matrix represent the fraction of the zone's total VAV output
  that each diffuser CV handles (summing to 1.0 per zone).

  Args:
    diffuser_matrix_to_fill (np.ndarray): A NumPy array (usually of zeros)
      with the building's CV grid shape, to be populated with diffuser fractions.
    room_cv_indices_map (RoomIndicesDict): A dictionary mapping room/zone names
      to sequences of (row, col) CV coordinates belonging to that room.
    interior_walls_map (building_utils.InteriorWalls): A binary map indicating
      interior wall locations, used to avoid placing diffusers in walls.
    diffuser_spacing_cvs (int): Desired spacing between diffusers, in CV units.
    buffer_from_walls_cvs (int): Minimum distance (in CVs) from a wall to a
      diffuser.

  Returns:
    np.ndarray: The `diffuser_matrix_to_fill` populated with diffuser fractions.
  """
  for room_name, cv_coords_list in room_cv_indices_map.items():
    if not room_name.startswith(constants.ROOM_STRING_DESIGNATOR):
      continue # Skip non-room entries if any

    # Determine diffuser locations for the current room
    diffuser_indices_in_room = thermal_diffuser_utils.diffuser_allocation_switch(
        room_cv_indices=cv_coords_list,
        spacing=diffuser_spacing_cvs,
        interior_walls=interior_walls_map, # Pass the map of interior walls
        buffer_from_walls=buffer_from_walls_cvs,
    )
    num_diffusers_in_room = len(diffuser_indices_in_room)
    if num_diffusers_in_room > 0:
      contribution_per_diffuser = 1.0 / float(num_diffusers_in_room)
      for r_idx, c_idx in diffuser_indices_in_room:
        # Ensure indices are within bounds, though diffuser_allocation_switch should handle this.
        if 0 <= r_idx < diffuser_matrix_to_fill.shape[0] and \
           0 <= c_idx < diffuser_matrix_to_fill.shape[1]:
          diffuser_matrix_to_fill[r_idx, c_idx] = contribution_per_diffuser
        else:
          logging.warning("Diffuser index (%d, %d) out of bounds for room %s.",
                          r_idx, c_idx, room_name)
  return diffuser_matrix_to_fill


class BaseSimulatorBuilding(abc.ABC):
  """Abstract base class for building models used within the simulator.

  Defines the common interface that different building representations
  (e.g., grid-based, floor-plan-based) must adhere to for compatibility
  with the `Simulator`.
  """

  @abc.abstractmethod
  def reset(self) -> None:
    """Resets the building's thermal state to initial conditions."""

  @abc.abstractmethod
  def get_zone_average_temps(self) -> Dict[Union[Tuple[int, int], str], float]:
    """Calculates and returns the average temperature for each zone.

    Returns:
      Dict[Union[Tuple[int, int], str], float]: A dictionary mapping zone
      identifiers (either (row, col) tuples for grid layout or string names
      for floor plan layout) to their average temperatures in Kelvin.
    """

  @property
  @abc.abstractmethod
  def density(self) -> np.ndarray:
    """np.ndarray: 2D grid of material densities (kg/m^3) for each CV."""

  @property
  @abc.abstractmethod
  def heat_capacity(self) -> np.ndarray:
    """np.ndarray: 2D grid of specific heat capacities (J/kgK) for each CV."""

  @property
  @abc.abstractmethod
  def conductivity(self) -> np.ndarray:
    """np.ndarray: 2D grid of thermal conductivities (W/mK) for each CV."""

  @property
  @abc.abstractmethod
  def cv_type(self) -> np.ndarray:
    """np.ndarray: 2D grid indicating the type of each CV (e.g., air, wall)."""


@gin.configurable
class Building(BaseSimulatorBuilding):
  """Represents a building as a grid of rooms with Control Volumes (CVs).

  This class models a building with a regular rectangular grid of rooms.
  Each CV has uniform size. It manages the thermal properties (temperature,
  conductivity, etc.) of these CVs and provides methods for interacting with
  them, such as applying heat and calculating zone temperatures.

  This class is an older implementation and might be superseded by
  `FloorPlanBasedBuilding` for more complex geometries.

  Attributes:
    cv_size_cm (float): Side length of a cubic Control Volume in centimeters.
    floor_height_cm (float): Height of rooms in centimeters.
    room_shape (Shape2D): (rows, cols) of air CVs within one room.
    building_shape (Shape2D): (rows, cols) of rooms in the building grid.
    temp (np.ndarray): 2D array of current temperatures (K) for each CV.
    input_q (np.ndarray): 2D array of heat input rates (W) to each CV.
    diffusers (np.ndarray): 2D array indicating diffuser locations and
      fractions.
    neighbors (List[List[List[Coordinates2D]]]): Nested list structure where
      `neighbors[r][c]` is a list of (nr, nc) coordinates of neighbors for
      CV at (r,c).
    _conductivity (np.ndarray): Private attribute for conductivity.
    _heat_capacity (np.ndarray): Private attribute for heat capacity.
    _density (np.ndarray): Private attribute for density.
    _initial_temp (float): Initial temperature (K) for all CVs at reset.
  """

  def __init__(
      self,
      cv_size_cm: float,
      floor_height_cm: float,
      room_shape: Shape2D,
      building_shape: Shape2D,
      initial_temp: float,
      inside_air_properties: MaterialProperties,
      inside_wall_properties: MaterialProperties,
      building_exterior_properties: MaterialProperties,
      deprecation: bool = False, # pylint: disable=unused-argument
  ):
    """Initializes a grid-based Building model.

    Args:
      cv_size_cm (float): Side length of a cubic Control Volume in cm.
      floor_height_cm (float): Height of the rooms in cm.
      room_shape (Shape2D): Tuple (rows, cols) defining the number of air
        CVs within a single room.
      building_shape (Shape2D): Tuple (rows, cols) defining the number of
        rooms in the building layout.
      initial_temp (float): Initial temperature in Kelvin for all CVs.
      inside_air_properties (MaterialProperties): Thermophysical properties for
        CVs representing interior air.
      inside_wall_properties (MaterialProperties): Thermophysical properties for
        CVs representing interior walls.
      building_exterior_properties (MaterialProperties): Thermophysical
        properties for CVs representing exterior building elements.
      deprecation (bool): Deprecated argument, no longer used.
    """
    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self.room_shape = room_shape
    self.building_shape = building_shape
    self._initial_temp = initial_temp

    # Calculate total matrix dimensions including walls
    # Each room adds (dim + 1 wall). Building has 2 outer wall layers + 1 start.
    nrows = (self.room_shape[0] + 1) * self.building_shape[0] + 3
    ncols = (self.room_shape[1] + 1) * self.building_shape[1] + 3
    matrix_shape = (nrows, ncols)

    # Initialize property matrices
    self._conductivity = np.full(matrix_shape, inside_air_properties.conductivity)
    assign_interior_wall_values(
        self._conductivity, inside_wall_properties.conductivity, self.room_shape
    )
    assign_building_exterior_values(
        self._conductivity, building_exterior_properties.conductivity
    )

    self._heat_capacity = np.full(matrix_shape, inside_air_properties.heat_capacity)
    assign_interior_wall_values(
        self._heat_capacity, inside_wall_properties.heat_capacity, self.room_shape
    )
    assign_building_exterior_values(
        self._heat_capacity, building_exterior_properties.heat_capacity
    )

    self._density = np.full(matrix_shape, inside_air_properties.density)
    assign_interior_wall_values(
        self._density, inside_wall_properties.density, self.room_shape
    )
    assign_building_exterior_values(
        self._density, building_exterior_properties.density
    )

    self.diffusers = generate_thermal_diffusers(matrix_shape, self.room_shape)
    self.neighbors = self._calculate_neighbors(matrix_shape)
    self.temp = np.array([]) # Initialized in reset
    self.input_q = np.array([]) # Initialized in reset
    self.reset()

  @property
  def density(self) -> np.ndarray:
    """np.ndarray: 2D grid of material densities (kg/m^3) for each CV."""
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    """np.ndarray: 2D grid of specific heat capacities (J/kgK) for each CV."""
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    """np.ndarray: 2D grid of thermal conductivities (W/mK) for each CV."""
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    """np.ndarray: 2D grid indicating CV type. Not fully implemented here."""
    # This class doesn't explicitly define cv_type like FloorPlanBasedBuilding.
    # Behavior might be implicitly handled by how properties are assigned.
    raise NotImplementedError(
        "cv_type is not explicitly defined for the grid-based Building class."
        " Use FloorPlanBasedBuilding for explicit CV typing."
    )

  def reset(self) -> None:
    """Resets temperatures and heat inputs to initial states."""
    nrows = (self.room_shape[0] + 1) * self.building_shape[0] + 3
    ncols = (self.room_shape[1] + 1) * self.building_shape[1] + 3
    self.temp = np.full((nrows, ncols), self._initial_temp)
    self.input_q = np.full((nrows, ncols), 0.0)

  def _calculate_neighbors(
      self, shape: Shape2D
  ) -> List[List[List[Coordinates2D]]]:
    """Computes list of valid neighbor coordinates for each CV in a grid.

    Args:
      shape (Shape2D): The (rows, cols) of the CV matrix.

    Returns:
      List[List[List[Coordinates2D]]]: A 3D nested list where `output[r][c]`
      contains a list of (nr, nc) tuples for valid neighbors of CV (r,c).
    """
    neighbor_list_matrix = [
        [[] for _ in range(shape[1])] for _ in range(shape[0])
    ]
    for r_idx in range(shape[0]):
      for c_idx in range(shape[1]):
        possible_neighbors_coords = [
            (r_idx - 1, c_idx), (r_idx + 1, c_idx),
            (r_idx, c_idx - 1), (r_idx, c_idx + 1)
        ]
        for nr, nc in possible_neighbors_coords:
          if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            neighbor_list_matrix[r_idx][c_idx].append((nr, nc))
    return neighbor_list_matrix

  def get_zone_thermal_energy_rate(
      self, zone_coordinates: Coordinates2D
  ) -> float:
    """Calculates total heat input rate (W) for a specified zone.

    Sums `input_q` values for all air CVs within the given zone.

    Args:
      zone_coordinates (Coordinates2D): (row, col) index of the zone within
        the building's room grid.

    Returns:
      float: Total thermal power (Watts) being input to the zone.
    """
    min_r, max_r, min_c, max_c = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    # Sum input_q over the air CVs of the zone
    return np.sum(self.input_q[min_r : max_r + 1, min_c : max_c + 1])

  def get_zone_temp_stats(
      self, zone_coordinates: Coordinates2D
  ) -> Tuple[float, float, float]:
    """Calculates min, max, and mean temperature for air CVs in a zone.

    Args:
      zone_coordinates (Coordinates2D): (row, col) index of the zone.

    Returns:
      Tuple[float, float, float]: (min_temp_K, max_temp_K, mean_temp_K) for
      the air CVs in the specified zone.
    """
    min_r, max_r, min_c, max_c = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    zone_temps = self.temp[min_r : max_r + 1, min_c : max_c + 1]
    return np.min(zone_temps), np.max(zone_temps), np.mean(zone_temps)

  def get_zone_average_temps(self) -> Dict[Tuple[int, int], float]:
    """Computes average temperature for all zones in the building.

    Returns:
      Dict[Tuple[int, int], float]: A dictionary mapping zone (row, col)
      indices to their average air temperatures in Kelvin.
    """
    avg_temps_map: Dict[Tuple[int, int], float] = {}
    for r_zone_idx in range(self.building_shape[0]):
      for c_zone_idx in range(self.building_shape[1]):
        zone_coords = (r_zone_idx, c_zone_idx)
        _, _, avg_temp = self.get_zone_temp_stats(zone_coords)
        avg_temps_map[zone_coords] = avg_temp
    return avg_temps_map

  def apply_thermal_power_zone(
      self, zone_coordinates: Coordinates2D, power_watts: float
  ) -> None:
    """Applies thermal power to a zone, distributed among its diffusers.

    The total `power_watts` is distributed to the CVs within the specified
    zone according to the fractional contributions defined in `self.diffusers`.

    Args:
      zone_coordinates (Coordinates2D): (row, col) index of the target zone.
      power_watts (float): Total thermal power (Watts) to apply to the zone.
        Positive for heating, negative for cooling.
    """
    min_r, max_r, min_c, max_c = get_zone_bounds(
        zone_coordinates, self.room_shape
    )
    # Iterate only over the air CVs of the specified zone
    for r_cv_idx in range(min_r, max_r + 1):
      for c_cv_idx in range(min_c, max_c + 1):
        if self.diffusers[r_cv_idx, c_cv_idx] > 0.0:
          self.input_q[r_cv_idx, c_cv_idx] = (
              power_watts * self.diffusers[r_cv_idx, c_cv_idx]
          )


@gin.configurable
class FloorPlanBasedBuilding(BaseSimulatorBuilding):
  """Models a building based on a floor plan image and zone map.

  This class offers a more flexible way to define building geometry compared to
  the grid-based `Building` class. It uses NumPy arrays derived from image
  files (floor plan, zone map) to determine wall locations, zone boundaries,
  and CV types.

  Key steps in initialization include:
  - Reading floor plan and zone map data.
  - Processing this data to identify exterior walls, interior walls, and
    exterior space (ambient air).
  - Assigning thermophysical properties to CVs based on their type.
  - Locating thermal diffusers within each zone.
  - Calculating CV neighbor relationships.

  Attributes:
    cv_size_cm (float): Side length of a cubic CV in cm.
    floor_height_cm (float): Height of rooms in cm.
    floor_plan (np.ndarray): 2D array representing the building layout.
    temp (np.ndarray): Current temperatures (K) of CVs.
    input_q (np.ndarray): Heat input rates (W) to CVs.
    diffusers (np.ndarray): Diffuser locations and fractions.
    neighbors (List[List[List[Coordinates2D]]]): CV neighbor data.
    len_neighbors (np.ndarray): Number of neighbors for each CV.
    _conductivity (np.ndarray): Grid of conductivities (W/mK).
    _heat_capacity (np.ndarray): Grid of heat capacities (J/kgK).
    _density (np.ndarray): Grid of densities (kg/m^3).
    _cv_type (np.ndarray): Grid of CV type labels (str).
    _initial_temp (float): Initial temperature (K) for reset.
    _reset_temp_values (Optional[np.ndarray]): Specific temperatures for reset.
    _room_dict (RoomIndicesDict): Maps zone names to CV coordinates.
    _exterior_walls (np.ndarray): Map of exterior wall CVs.
    _interior_walls (np.ndarray): Map of interior wall CVs.
    _exterior_space (np.ndarray): Map of exterior space CVs.
    _zone_map (np.ndarray): Map defining zone boundaries.
    _convection_simulator (Optional[base_convection_simulator.BaseConvectionSimulator]):
      Simulator for air convection effects.
  """

  def __init__(
      self,
      cv_size_cm: float,
      floor_height_cm: float,
      initial_temp: float,
      inside_air_properties: MaterialProperties,
      inside_wall_properties: MaterialProperties,
      building_exterior_properties: MaterialProperties,
      zone_map: Optional[np.ndarray] = None,
      zone_map_filepath: Optional[str] = None,
      floor_plan: Optional[np.ndarray] = None,
      floor_plan_filepath: Optional[str] = None,
      buffer_from_walls: int = 3,
      convection_simulator: Optional[
          base_convection_simulator.BaseConvectionSimulator
      ] = None,
      reset_temp_values: Optional[np.ndarray] = None,
  ):
    """Initializes a FloorPlanBasedBuilding model.

    Args:
      cv_size_cm (float): Side length of a cubic Control Volume in cm.
      floor_height_cm (float): Height of the rooms in cm.
      initial_temp (float): Default initial temperature (K) for all CVs if
        `reset_temp_values` is not provided.
      inside_air_properties (MaterialProperties): Properties for interior air.
      inside_wall_properties (MaterialProperties): Properties for interior walls.
      building_exterior_properties (MaterialProperties): Properties for
        exterior building elements.
      zone_map (Optional[np.ndarray]): A 2D NumPy array where each unique
        non-zero value identifies a distinct zone. If None,
        `zone_map_filepath` must be provided.
      zone_map_filepath (Optional[str]): Path to a file (e.g., .npy)
        containing the `zone_map` array.
      floor_plan (Optional[np.ndarray]): A 2D NumPy array representing the
        building layout (e.g., values indicating air, interior wall, exterior
        wall). If None, `floor_plan_filepath` must be provided.
      floor_plan_filepath (Optional[str]): Path to a file containing the
        `floor_plan` array.
      buffer_from_walls (int): Minimum distance (in CVs) to maintain between
        thermal diffusers and walls.
      convection_simulator (Optional[base_convection_simulator.BaseConvectionSimulator]):
        An optional simulator for modeling air convection effects within rooms.
      reset_temp_values (Optional[np.ndarray]): A 2D NumPy array of specific
        temperatures (K) to use when resetting the building state. If None,
        `initial_temp` is used uniformly.

    Raises:
      ValueError: If essential inputs (like floor plan or zone map, either
        directly or via filepath) are missing or provided ambiguously.
    """
    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self._initial_temp = initial_temp
    self._convection_simulator = convection_simulator
    self._reset_temp_values = reset_temp_values

    # Load floor plan
    if floor_plan is None and floor_plan_filepath is None:
      raise ValueError("Either floor_plan or floor_plan_filepath must be provided.")
    if floor_plan is not None and floor_plan_filepath is not None:
      raise ValueError("Provide either floor_plan or floor_plan_filepath, not both.")
    self.floor_plan = floor_plan if floor_plan is not None else \
                      building_utils.read_floor_plan_from_filepath(floor_plan_filepath)

    # Load zone map
    if zone_map is None and zone_map_filepath is None:
      raise ValueError("Either zone_map or zone_map_filepath must be provided.")
    if zone_map is not None and zone_map_filepath is not None:
      raise ValueError("Provide either zone_map or zone_map_filepath, not both.")
    self._zone_map = zone_map if zone_map is not None else \
                     building_utils.read_floor_plan_from_filepath(zone_map_filepath)

    # Process floor plan and zone map to derive building structure
    (self._room_dict, exterior_walls_raw, interior_walls_raw, self._exterior_space) = (
        building_utils.construct_building_data_types(
            floor_plan=self.floor_plan, zone_map=self._zone_map
        )
    )
    self._exterior_walls, self._interior_walls = enlarge_exterior_walls(
        exterior_walls=exterior_walls_raw, interior_walls=interior_walls_raw
    )

    # Assign thermophysical properties based on CV types
    self._conductivity = _assign_interior_and_exterior_values(
        self._exterior_walls, self._interior_walls,
        inside_wall_properties.conductivity,
        building_exterior_properties.conductivity,
        inside_air_properties.conductivity,
    )
    self._heat_capacity = _assign_interior_and_exterior_values(
        self._exterior_walls, self._interior_walls,
        inside_wall_properties.heat_capacity,
        building_exterior_properties.heat_capacity,
        inside_air_properties.heat_capacity,
    )
    self._density = _assign_interior_and_exterior_values(
        self._exterior_walls, self._interior_walls,
        inside_wall_properties.density,
        building_exterior_properties.density,
        inside_air_properties.density,
    )

    # Initialize and assign thermal diffusers
    self.diffusers = np.zeros_like(self._exterior_walls, dtype=float)
    self.diffusers = _assign_thermal_diffusers(
        self.diffusers, self._room_dict, self._interior_walls,
        buffer_from_walls=buffer_from_walls
    )

    # Determine CV types (exterior space, interior space, wall)
    self._cv_type = _construct_cv_type_array(self._exterior_walls, self._exterior_space)

    # Calculate neighbor relationships for FDM
    self.neighbors = self._calculate_neighbors()
    self.len_neighbors = self._calculate_length_of_neighbors()

    self.temp = np.array([]) # Initialized in reset
    self.input_q = np.array([]) # Initialized in reset
    self.reset()

  @property
  def density(self) -> np.ndarray:
    """np.ndarray: 2D grid of material densities (kg/m^3) for each CV."""
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    """np.ndarray: 2D grid of specific heat capacities (J/kgK) for each CV."""
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    """np.ndarray: 2D grid of thermal conductivities (W/mK) for each CV."""
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    """np.ndarray: 2D grid of CV type labels (str), e.g., 'exterior_space'."""
    return self._cv_type

  def reset(self) -> None:
    """Resets building temperatures and heat inputs to initial states.

    If `_reset_temp_values` was provided during initialization, those specific
    temperatures are used; otherwise, `_initial_temp` is applied uniformly.
    """
    if self._reset_temp_values is not None:
      if self._reset_temp_values.shape == self._exterior_walls.shape:
        self.temp = np.copy(self._reset_temp_values)
      else:
        logging.warning(
            "Shape of reset_temp_values %s does not match building shape %s. "
            "Using uniform initial_temp instead.",
            self._reset_temp_values.shape, self._exterior_walls.shape
        )
        self.temp = np.full_like(self._exterior_walls, self._initial_temp, dtype=float)
    else:
      self.temp = np.full_like(self._exterior_walls, self._initial_temp, dtype=float)
    self.input_q = np.zeros_like(self._exterior_walls, dtype=float)

  def _calculate_neighbors(self) -> List[List[List[Coordinates2D]]]:
    """Computes valid neighbor coordinates for each CV in the building grid.

    A CV's neighbors are its adjacent CVs (up, down, left, right) that are
    not exterior space. This is used for heat conduction calculations in FDM.

    Returns:
      List[List[List[Coordinates2D]]]: A 3D nested list where `output[r][c]`
      contains a list of (nr, nc) tuples for valid neighbors of CV (r,c).
    """
    shape = self._exterior_walls.shape
    neighbor_list_matrix = [
        [[] for _ in range(shape[1])] for _ in range(shape[0])
    ]
    for r_idx in range(shape[0]):
      for c_idx in range(shape[1]):
        # CVs that are exterior space do not have internal neighbors for FDM
        if self.cv_type[r_idx, c_idx] == constants.LABEL_FOR_EXTERIOR_SPACE:
          continue

        possible_neighbors_coords = [
            (r_idx - 1, c_idx), (r_idx + 1, c_idx),
            (r_idx, c_idx - 1), (r_idx, c_idx + 1)
        ]
        for nr, nc in possible_neighbors_coords:
          # Check bounds and ensure neighbor is not exterior space
          if (0 <= nr < shape[0] and 0 <= nc < shape[1] and
              self.cv_type[nr, nc] != constants.LABEL_FOR_EXTERIOR_SPACE):
            neighbor_list_matrix[r_idx][c_idx].append((nr, nc))
    return neighbor_list_matrix

  def _calculate_length_of_neighbors(self) -> np.ndarray:
    """Creates a 2D array storing the number of valid neighbors for each CV."""
    len_neighbors_matrix = np.full_like(self._exterior_walls, 0, dtype=int)
    for r_idx in range(len_neighbors_matrix.shape[0]):
      for c_idx in range(len_neighbors_matrix.shape[1]):
        len_neighbors_matrix[r_idx, c_idx] = len(self.neighbors[r_idx][c_idx])
    return len_neighbors_matrix

  def get_zone_thermal_energy_rate(self, zone_name: str) -> float:
    """Calculates total heat input rate (W) for a specified zone.

    Args:
      zone_name (str): The string identifier of the zone (must be a key in
        `self._room_dict`).

    Returns:
      float: Total thermal power (Watts) being input to the zone's CVs.

    Raises:
      ValueError: If `zone_name` is not found in `self._room_dict`.
    """
    if zone_name not in self._room_dict:
      raise ValueError(f"Zone '{zone_name}' not found in room dictionary.")

    zone_cv_coords = self._room_dict[zone_name]
    # Sum input_q for all CVs belonging to this zone
    total_q_watts = sum(self.input_q[r, c] for r, c in zone_cv_coords)
    return total_q_watts

  def get_zone_temp_stats(self, zone_name: str) -> Tuple[float, float, float]:
    """Calculates min, max, and mean temperature for CVs in a specified zone.

    Args:
      zone_name (str): The string identifier of the zone.

    Returns:
      Tuple[float, float, float]: (min_temp_K, max_temp_K, mean_temp_K) for
      the CVs in the specified zone. Returns (NaN, NaN, NaN) if zone is empty.

    Raises:
      ValueError: If `zone_name` is not found in `self._room_dict`.
    """
    if zone_name not in self._room_dict:
      raise ValueError(f"Zone '{zone_name}' not found in room dictionary.")

    zone_cv_coords = self._room_dict[zone_name]
    if not zone_cv_coords: # Handle case of an empty zone
        return (np.nan, np.nan, np.nan)
    zone_temps = [self.temp[r, c] for r, c in zone_cv_coords]
    return np.min(zone_temps), np.max(zone_temps), np.mean(zone_temps)

  def get_zone_average_temps(self) -> Dict[str, float]:
    """Computes average temperature for all defined rooms/zones.

    Returns:
      Dict[str, float]: A dictionary mapping zone name (str) to its
      average air temperature in Kelvin.
    """
    avg_temps_map: Dict[str, float] = {}
    for zone_name_key in self._room_dict:
      # Ensure we only process actual rooms/zones based on naming convention
      if zone_name_key.startswith(constants.ROOM_STRING_DESIGNATOR):
        _, _, avg_temp = self.get_zone_temp_stats(zone_name_key)
        if not np.isnan(avg_temp): # Only add if stats are valid
             avg_temps_map[zone_name_key] = avg_temp
    return avg_temps_map

  def apply_thermal_power_zone(self, zone_name: str, power_watts: float) -> None:
    """Applies thermal power to a zone, distributed among its diffusers.

    Args:
      zone_name (str): The string identifier of the target zone.
      power_watts (float): Total thermal power (Watts) to apply. Positive for
        heating, negative for cooling.

    Raises:
      ValueError: If `zone_name` is not found in `self._room_dict`.
    """
    if zone_name not in self._room_dict:
      raise ValueError(f"Zone '{zone_name}' not found in room dictionary.")

    zone_cv_coords = self._room_dict[zone_name]
    for r_cv_idx, c_cv_idx in zone_cv_coords:
      diffuser_fraction = self.diffusers[r_cv_idx, c_cv_idx]
      if diffuser_fraction > 0.0:
        # Apply proportional power to this diffuser CV
        self.input_q[r_cv_idx, c_cv_idx] = power_watts * diffuser_fraction

  def apply_convection(self) -> None:
    """Applies air convection effects if a convection simulator is configured.

    This method delegates to an optional `BaseConvectionSimulator` to model
    air movement and its impact on CV temperatures within rooms.
    """
    if self._convection_simulator is not None:
      self._convection_simulator.apply_convection(self._room_dict, self.temp)
