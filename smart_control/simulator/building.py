"""Defines classes and functions for representing building thermal properties.

This module includes data structures for material properties, and classes that
represent the building's physical structure as a grid of control volumes (CVs).
It provides utilities for initializing this grid based on room shapes or detailed
floor plans, assigning thermal properties (conductivity, heat capacity, density)
to different CVs (air, walls, exterior), calculating CV neighbors, and managing
thermal inputs (e.g., from HVAC diffusers).

Two main building representations are provided:
- `Building`: A simpler model for buildings with regular, grid-aligned rectangular rooms.
  (Note: This class is marked with a deprecation comment in the code, suggesting
  `FloorPlanBasedBuilding` is preferred).
- `FloorPlanBasedBuilding`: A more flexible model that constructs the thermal
  representation from a detailed floor plan and zone map, allowing for complex
  room geometries.

Both inherit from `BaseSimulatorBuilding`, an abstract class defining the common
interface for building models used within the thermal simulator.
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

# Type alias for 2D coordinates (row, column)
Coordinates2D = Tuple[int, int]
# Type alias for 2D shape (number of rows, number of columns)
Shape2D = Tuple[int, int]
# Type alias for a dictionary mapping room names (str) to a sequence of their constituent CV coordinates.
RoomIndicesDict = Dict[str, Sequence[Coordinates2D]]


@gin.configurable
@dataclasses.dataclass
class MaterialProperties:
  """Holds the thermal physical properties for a material.

  These properties are used to define the characteristics of different components
  (air, walls, etc.) within the building simulation grid.

  Attributes:
    conductivity: Thermal conductivity of the material (in Watts per meter-Kelvin,
      W/m/K). This measures the material's ability to conduct heat.
    heat_capacity: Specific heat capacity of the material (in Joules per
      kilogram-Kelvin, J/kg/K). This is the amount of heat needed to raise the
      temperature of 1 kg of the material by 1 Kelvin.
    density: Density of the material (in kilograms per cubic meter, kg/m^3).
  """
  conductivity: float
  heat_capacity: float
  density: float


def _check_room_sizes(matrix_shape: Shape2D, room_shape: Shape2D) -> None:
  """Validates if room dimensions are compatible with the overall matrix shape.

  This helper function is used by the deprecated `Building` class which assumes
  a grid of rectangular rooms separated by single-CV-thick walls, with a
  2-CV-thick exterior boundary. It checks if the `matrix_shape` can be
  correctly divided into rooms of `room_shape` plus walls.

  Args:
    matrix_shape: A 2-tuple `(rows, cols)` representing the shape of the
      overall building matrix.
    room_shape: A 2-tuple `(room_rows, room_cols)` representing the number of
      air control volumes (CVs) in each dimension within a single room.

  Raises:
    ValueError: If `room_shape` dimensions do not fit cleanly into the
      `matrix_shape` according to the assumed layout (i.e., if
      `(matrix_dim - 3) % (room_dim + 1)` is not zero for either dimension).
  """
  if (matrix_shape[0] - 3) % (room_shape[0] + 1) != 0:
    raise ValueError(
        f"Room shape dimension {room_shape[0]} is not compatible with matrix"
        f" dimension {matrix_shape[0]} given wall/boundary assumptions."
    )
  if (matrix_shape[1] - 3) % (room_shape[1] + 1) != 0:
    raise ValueError(
        f"Room shape dimension {room_shape[1]} is not compatible with matrix"
        f" dimension {matrix_shape[1]} given wall/boundary assumptions."
    )


def assign_building_exterior_values(array: np.ndarray, value: float) -> None:
  """Assigns a specified value to the exterior boundary of a building grid.

  This function modifies the input `array` in place. It sets the outermost two
  layers of cells on all sides of the matrix to the given `value`. These layers
  typically represent the building's thick exterior walls and the adjacent
  ambient air in grid-based thermal simulations.

  Args:
    array: The NumPy array (representing a building grid property like
      conductivity or temperature) to be modified.
    value: The float value to assign to the exterior boundary cells.
  """
  array[:, [0, 1, -2, -1]] = value  # Columns 0, 1, second-to-last, last
  array[[0, 1, -2, -1], :] = value  # Rows 0, 1, second-to-last, last


def assign_interior_wall_values(
    array: np.ndarray, value: float, room_shape: Shape2D
) -> None:
  """Assigns a specified value to interior wall locations in a building grid.

  This function modifies the input `array` in place. It identifies locations
  that correspond to interior walls based on a regular grid of rooms (defined
  by `room_shape`) and assigns `value` to these locations. This is primarily
  used by the deprecated `Building` class.

  Args:
    array: The NumPy array (e.g., for conductivity) to be modified.
    value: The float value to assign to interior wall cells.
    room_shape: A 2-tuple `(room_rows, room_cols)` defining the dimensions of
      air CVs within each room. This is used to infer wall locations in a
      grid layout.
  """
  _check_room_sizes(array.shape, room_shape)
  nrows, ncols = array.shape

  # Assign to horizontal interior walls
  for x in range(room_shape[0] + 2, nrows - 2, room_shape[0] + 1):
    array[x, 2 : ncols - 2] = value # Fill entire row segment for wall
  # Assign to vertical interior walls
  for y in range(room_shape[1] + 2, ncols - 2, room_shape[1] + 1):
    array[2 : nrows - 2, y] = value # Fill entire col segment for wall


def generate_thermal_diffusers(
    matrix_shape: Shape2D, room_shape: Shape2D
) -> np.ndarray:
  """Generates a grid representing thermal diffuser placements for rectangular rooms.

  This function creates a NumPy array where non-zero values indicate the
  location and strength of thermal diffusers. For each room in a regular grid
  layout (defined by `matrix_shape` and `room_shape`), it places four diffusers,
  aiming for even distribution. The value at each diffuser location represents
  its proportion of the total thermal power supplied to that room (summing to
  1.0 per room if four diffusers are placed).

  This is primarily used by the deprecated `Building` class.

  Args:
    matrix_shape: A 2-tuple `(rows, cols)` for the overall building grid.
    room_shape: A 2-tuple `(room_rows, room_cols)` for the air CV dimensions
      within each room.

  Returns:
    A NumPy array of dtype float32 and shape `matrix_shape`, with non-zero
    values at diffuser locations.
  """
  _check_room_sizes(matrix_shape, room_shape)

  n_diffusers_per_dim = 2  # Assumes 2x2 diffusers per room
  diffuser_value = 1.0 / (n_diffusers_per_dim**2) # Each diffuser gets 1/4 of power

  diffusers = np.zeros(shape=matrix_shape, dtype=np.float32)
  nrows, ncols = matrix_shape

  # Calculate spacing for diffusers within a room
  # empty_spaces_x is num air CVs in room width minus num diffusers in that dim
  empty_spaces_x = room_shape[0] - n_diffusers_per_dim
  # Distribute empty spaces: before first diffuser, between diffusers, after last
  diff_1_offset_x = empty_spaces_x // (n_diffusers_per_dim + 1)
  # diff_2_offset_x assumes symmetry or specific placement for the second diffuser
  diff_2_offset_x = room_shape[0] - 1 - diff_1_offset_x # Place from other end

  empty_spaces_y = room_shape[1] - n_diffusers_per_dim
  diff_1_offset_y = empty_spaces_y // (n_diffusers_per_dim + 1)
  diff_2_offset_y = room_shape[1] - 1 - diff_1_offset_y

  # Iterate through each room's top-left air CV starting coordinate
  for room_start_row in range(2, nrows - 3, room_shape[0] + 1):
    for room_start_col in range(2, ncols - 3, room_shape[1] + 1):
      # Place 4 diffusers in the current room
      diffusers[room_start_row + diff_1_offset_x, room_start_col + diff_1_offset_y] = diffuser_value
      diffusers[room_start_row + diff_2_offset_x, room_start_col + diff_1_offset_y] = diffuser_value
      diffusers[room_start_row + diff_1_offset_x, room_start_col + diff_2_offset_y] = diffuser_value
      diffusers[room_start_row + diff_2_offset_x, room_start_col + diff_2_offset_y] = diffuser_value
  return diffusers


def get_zone_bounds(
    zone_coordinates: Coordinates2D, room_shape: Shape2D
) -> Tuple[int, int, int, int]:
  """Calculates the 0-indexed min/max row/col bounds for air CVs in a zone.

  This helper is for the deprecated `Building` class's grid layout.
  It determines the bounding box of air control volumes for a zone specified
  by its `zone_coordinates` (e.g., (0,0) for the first room) within a building
  composed of rooms of `room_shape`. The calculation accounts for the
  2-CV exterior boundary and 1-CV interior walls.

  Args:
    zone_coordinates: A tuple `(zone_row_idx, zone_col_idx)` representing the
      0-indexed position of the zone in the building's grid of rooms.
    room_shape: A 2-tuple `(room_rows, room_cols)` for the air CV dimensions
      within each room.

  Returns:
    A tuple `(row_min, row_max, col_min, col_max)` representing the inclusive
    0-indexed bounds of the air CVs for the specified zone.
  """
  zone_x, zone_y = zone_coordinates
  # Start index of air CVs: 2 (for exterior boundary) + zone_idx * (room_dim + 1 wall)
  x_min = zone_x * (room_shape[0] + 1) + 2
  x_max = x_min + room_shape[0] - 1 # End index is start + room_dim - 1
  y_min = zone_y * (room_shape[1] + 1) + 2
  y_max = y_min + room_shape[1] - 1
  return (x_min, x_max, y_min, y_max)


#### Helper code below here marks the updated helper functions that Lucas wrote:
# (Docstrings for these functions will also be updated)

def enlarge_exterior_walls(
    exterior_walls: building_utils.ExteriorWalls,
    interior_walls: building_utils.InteriorWalls,
) -> Tuple[building_utils.ExteriorWalls, building_utils.InteriorWalls]:
  """Expands exterior wall regions and shrinks interior wall regions in masks.

  This function processes binary masks representing exterior and interior walls.
  It "thickens" the exterior walls by expanding their regions by a defined
  amount (`constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT`) and correspondingly
  "thins" the interior walls where they overlap with the expanded exterior.
  This is a utility for refining wall definitions derived from a floor plan.

  Args:
    exterior_walls: A NumPy array (typically `building_utils.ExteriorWalls` type)
      representing a binary mask of exterior wall locations.
    interior_walls: A NumPy array (typically `building_utils.InteriorWalls` type)
      representing a binary mask of interior wall locations.

  Returns:
    A tuple `(augmented_exterior_walls, shrunk_interior_walls)` containing the
    modified NumPy arrays for exterior and interior wall masks.
  """
  # Create copies to avoid modifying original arrays if they are passed by reference elsewhere
  exterior_walls_binary = exterior_walls.copy()
  interior_walls_binary = interior_walls.copy()

  # Ensure masks are binary (0 or 1) based on defined constant values
  exterior_walls_binary = np.uint8(
      exterior_walls_binary == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
  )
  interior_walls_binary = np.uint8(
      interior_walls_binary == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  )

  # Enlarge the exterior wall region
  exterior_walls_augmented_temp = building_utils.enlarge_component(
      exterior_walls_binary, constants.EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT
  )

  # Identify areas that are part of original walls or the newly expanded exterior region
  walls_or_expanded = (
      exterior_walls_augmented_temp
      + interior_walls_binary
      + exterior_walls_binary # Add original exterior walls back to ensure they are included
  )
  # Augmented exterior walls are those that were originally exterior or became part of the expansion
  exterior_walls_augmented = np.int16(
      walls_or_expanded >= constants.WALLS_AND_EXPANDED_BOOLS # Check if it's any kind of wall/expansion
  ) * constants.EXTERIOR_WALL_VALUE_IN_FUNCTION # Assign exterior wall value

  # Shrink interior walls: an interior wall remains only if it doesn't overlap
  # with the (newly augmented) exterior walls.
  interior_walls_shrunk = np.int16(
      (interior_walls + exterior_walls_augmented) == constants.INTERIOR_WALL_VALUE_IN_FUNCTION
  ) * constants.INTERIOR_WALL_VALUE_IN_FUNCTION

  return exterior_walls_augmented, interior_walls_shrunk


def _assign_interior_and_exterior_values(
    exterior_walls: np.ndarray,
    interior_walls: np.ndarray,
    interior_wall_value: float,
    exterior_wall_value: float,
    interior_and_exterior_space_value: float,
) -> np.ndarray:
  """Assigns material property values to a grid based on wall masks.

  This function populates a NumPy array (e.g., for conductivity, density)
  by assigning specific values to locations identified as interior walls,
  exterior walls, or general space (neither wall type). It uses pre-processed
  binary masks for exterior and interior walls.

  Args:
    exterior_walls: A NumPy array (binary mask) where non-zero values (matching
      `constants.EXTERIOR_WALL_VALUE_IN_FUNCTION`) indicate exterior walls.
    interior_walls: A NumPy array (binary mask) where non-zero values (matching
      `constants.INTERIOR_WALL_VALUE_IN_FUNCTION`) indicate interior walls.
    interior_wall_value: The property value to assign to interior wall locations.
    exterior_wall_value: The property value to assign to exterior wall locations.
    interior_and_exterior_space_value: The property value to assign to all
      other locations (considered air or general space).

  Returns:
    A new NumPy array populated with the assigned property values.
  """
  # Uses nested np.where for conditional assignment:
  # 1. If interior_walls is true, assign interior_wall_value.
  # 2. Else, if exterior_walls is true, assign exterior_wall_value.
  # 3. Else (neither wall type), assign interior_and_exterior_space_value.
  array_to_return = np.where(
      interior_walls == constants.INTERIOR_WALL_VALUE_IN_FUNCTION,
      interior_wall_value,
      np.where(
          exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION,
          exterior_wall_value,
          interior_and_exterior_space_value,
      ),
  )
  return array_to_return


def _construct_cv_type_array(
    exterior_walls: np.ndarray, exterior_space: np.ndarray
) -> np.ndarray:
  """Creates a matrix identifying the type of each Control Volume (CV).

  This function generates a NumPy array where each cell is labeled with a string
  constant indicating its type: exterior space (outside air), interior space
  (room air), or wall. This pre-calculated CV type matrix can be used by other
  simulation components to apply type-specific logic.

  Args:
    exterior_walls: A NumPy array (binary mask) identifying exterior wall
      locations (e.g., using `constants.EXTERIOR_WALL_VALUE_IN_FUNCTION`).
      In this context, it seems to be used inversely: locations *not*
      `constants.INTERIOR_SPACE_VALUE_IN_FUNCTION` (when not exterior_space)
      are considered walls. This logic might need careful review based on how
      the input masks are generated.
    exterior_space: A NumPy array (binary mask) identifying exterior space
      (outside air) locations (e.g., using `constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION`).

  Returns:
    A NumPy array of strings, where each string is a CV type label from
    `smart_control.simulator.constants` (e.g., `LABEL_FOR_EXTERIOR_SPACE`).
  """
  # Logic:
  # 1. If exterior_space is true, label as LABEL_FOR_EXTERIOR_SPACE.
  # 2. Else, if exterior_walls indicates INTERIOR_SPACE_VALUE_IN_FUNCTION (this seems counterintuitive
  #    for a variable named exterior_walls, suggesting it might be a general space mask excluding exterior),
  #    label as LABEL_FOR_INTERIOR_SPACE.
  # 3. Else, label as LABEL_FOR_WALLS.
  # This interpretation depends heavily on the precise meaning of the input masks.
  return np.where(
      exterior_space == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
      constants.LABEL_FOR_EXTERIOR_SPACE,
      np.where(
          exterior_walls == constants.INTERIOR_SPACE_VALUE_IN_FUNCTION, # This condition might imply 'exterior_walls' is actually an 'is_not_wall_and_not_exterior_space' mask.
          constants.LABEL_FOR_INTERIOR_SPACE,
          constants.LABEL_FOR_WALLS,
      ),
  )


def _assign_thermal_diffusers(
    array_to_fill: np.ndarray,
    room_dict: RoomIndicesDict,
    interior_walls: building_utils.InteriorWalls,
    diffuser_spacing: int = 10,
    buffer_from_walls: int = 5,
) -> np.ndarray:
  """Places thermal diffuser locations within rooms on a grid.

  This function populates `array_to_fill` with diffuser strength values (where
  the sum of strengths per room is 1.0). It uses `thermal_diffuser_utils`
  to determine diffuser coordinates within each room defined in `room_dict`,
  considering `diffuser_spacing` and `buffer_from_walls`. This method is
  designed to handle complex room geometries.

  Args:
    array_to_fill: A NumPy array (typically pre-filled with zeros) of the same
      shape as the building grid, which will be populated with diffuser values.
    room_dict: A dictionary mapping room names (strings) to sequences of
      `(row, col)` coordinates defining the CVs within each room.
    interior_walls: A NumPy array (binary mask) of interior wall locations, used
      by `diffuser_allocation_switch` to avoid placing diffusers in walls.
    diffuser_spacing: Desired spacing (in CV units) between diffusers.
    buffer_from_walls: Minimum distance (in CV units) between diffusers and
      room walls.

  Returns:
    The modified `array_to_fill` with diffuser locations and their proportional
    strength values.
  """
  for key, value in room_dict.items():
    if not key.startswith(constants.ROOM_STRING_DESIGNATOR): # Process only designated rooms
      continue

    # Get diffuser coordinates for the current room
    inds = thermal_diffuser_utils.diffuser_allocation_switch(
        room_cv_indices=value,
        spacing=diffuser_spacing,
        interior_walls=interior_walls,
        buffer_from_walls=buffer_from_walls,
    )
    num_inds = len(inds)
    if num_inds > 0:
      diffuser_strength = 1.0 / float(num_inds)
      for ind in inds:
        array_to_fill[tuple(ind)] = diffuser_strength # Assign proportional strength
  return array_to_fill


class BaseSimulatorBuilding(abc.ABC):
  """Abstract base class for building models used in thermal simulations.

  This class defines the common interface that different building representations
  (e.g., simple grid-based, detailed floor-plan-based) must adhere to for use
  within the thermal simulation environment. It mandates methods for resetting
  the building state and properties for accessing its thermal characteristics.
  """

  @abc.abstractmethod
  def reset(self):
    """Resets the building's thermal state to its initial parameters.
    
    This typically involves setting temperatures and heat inputs to their
    starting values at the beginning of a simulation episode.
    """

  @abc.abstractmethod
  def get_zone_average_temps(
      self,
  ) -> Union[
      Dict[Tuple[int, int], Any], # For deprecated Building class (zone_coords as keys)
      Dict[str, Any],             # For FloorPlanBasedBuilding (zone_names as keys)
  ]:
    """Calculates and returns the average temperature for each defined zone.

    Returns:
      A dictionary where keys are zone identifiers (either coordinates or names)
      and values are the average temperatures (typically in Kelvin) of the
      control volumes within those zones.
    """

  @property
  @abc.abstractmethod
  def density(self) -> np.ndarray:
    """The NumPy array representing material densities (kg/m^3) for each CV."""

  @property
  @abc.abstractmethod
  def heat_capacity(self) -> np.ndarray:
    """The NumPy array representing specific heat capacities (J/kg/K) for each CV."""

  @property
  @abc.abstractmethod
  def conductivity(self) -> np.ndarray:
    """The NumPy array representing thermal conductivities (W/m/K) for each CV."""

  @property
  @abc.abstractmethod
  def cv_type(self) -> np.ndarray:
    """The NumPy array of strings labeling the type of each CV (e.g., air, wall)."""


@gin.configurable
class Building(BaseSimulatorBuilding):
  """Represents a building as a grid of rectangular rooms and control volumes.

  Note: This class is based on a simplified grid layout of identical rectangular
  rooms and is marked for deprecation. For more complex or realistic building
  geometries, `FloorPlanBasedBuilding` should be used.

  It manages the thermal state (temperature, heat input) and properties
  (conductivity, heat capacity, density) of each control volume (CV) in the grid.
  The building layout includes exterior walls, interior walls, and air spaces.

  Attributes:
    cv_size_cm: Size (width, length, height) of each CV in centimeters.
    floor_height_cm: Floor-to-ceiling height of rooms in centimeters.
    room_shape: Tuple `(rows, cols)` defining air CVs per room.
    building_shape: Tuple `(rows, cols)` defining number of rooms.
    temp: NumPy array of current temperatures (K) for each CV.
    conductivity: NumPy array of thermal conductivities (W/m/K) for each CV.
    heat_capacity: NumPy array of specific heat capacities (J/kg/K) for each CV.
    density: NumPy array of material densities (kg/m^3) for each CV.
    input_q: NumPy array of heat energy rates (Watts) applied to each CV.
    diffusers: NumPy array indicating diffuser locations and strengths.
    neighbors: Matrix (list of lists of lists) containing coordinates of
      neighboring CVs for each CV.
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
      deprecation: bool = False, # Kept for gin compatibility if used
  ):
    """Initializes a grid-based Building model.

    This constructor sets up matrices for thermal properties (conductivity,
    heat capacity, density), diffuser locations, and neighbor relationships based
    on the specified room and building shapes.

    Args:
      cv_size_cm: The size (width, length, and height assumed equal) of each
        cubic control volume in centimeters.
      floor_height_cm: The height from floor to ceiling for each room in
        centimeters. (Note: `cv_size_cm` might imply CV height too).
      room_shape: A 2-tuple `(room_rows, room_cols)` representing the number of
        air control volumes in the width and length dimensions of each room.
      building_shape: A 2-tuple `(num_rooms_rows, num_rooms_cols)` representing
        the number of rooms in the building's grid layout.
      initial_temp: The initial temperature (in Kelvin) to assign to all
        control volumes at the start of a simulation.
      inside_air_properties: A `MaterialProperties` object defining the thermal
        properties of interior air.
      inside_wall_properties: A `MaterialProperties` object for interior walls.
      building_exterior_properties: A `MaterialProperties` object for the
        building's exterior (walls and potentially ambient air representation).
      deprecation: A flag indicating if the deprecated simple grid logic should
        be used. If False (default for this class as originally intended),
        it initializes the grid. If True, it might behave differently or skip
        initialization if `FloorPlanBasedBuilding` is the intended path.
        (Original code comment: "TODO(sipple): delete the class when deprecation
        is finished.")
    """
    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self.room_shape = room_shape
    self.building_shape = building_shape
    self._initial_temp = initial_temp

    # The 'deprecation' flag's logic seems inverted in the original code for this class.
    # If not deprecation (i.e. use this class's logic):
    if not deprecation:
      nrows = (self.room_shape[0] + 1) * self.building_shape[0] + 3
      ncols = (self.room_shape[1] + 1) * self.building_shape[1] + 3
      matrix_shape = (nrows, ncols)

      self._conductivity = np.full(matrix_shape, inside_air_properties.conductivity)
      assign_interior_wall_values(self._conductivity, inside_wall_properties.conductivity, self.room_shape)
      assign_building_exterior_values(self._conductivity, building_exterior_properties.conductivity)

      self._heat_capacity = np.full(matrix_shape, inside_air_properties.heat_capacity)
      assign_interior_wall_values(self._heat_capacity, inside_wall_properties.heat_capacity, self.room_shape)
      assign_building_exterior_values(self._heat_capacity, building_exterior_properties.heat_capacity)

      self._density = np.full(matrix_shape, inside_air_properties.density)
      assign_interior_wall_values(self._density, inside_wall_properties.density, self.room_shape)
      assign_building_exterior_values(self._density, building_exterior_properties.density)

      self.diffusers = generate_thermal_diffusers(matrix_shape, self.room_shape)
      self.neighbors = self._calculate_neighbors(matrix_shape)
      self.reset() # Initialize temp and input_q
    else:
      # If 'deprecation' is True, this simpler Building model might not fully initialize,
      # deferring to FloorPlanBasedBuilding. Initialize core attributes to prevent errors.
      self._conductivity = np.array([])
      self._heat_capacity = np.array([])
      self._density = np.array([])
      self.diffusers = np.array([])
      self.neighbors = []
      self.temp = np.array([])
      self.input_q = np.array([])


  @property
  def density(self) -> np.ndarray:
    """The NumPy array representing material densities (kg/m^3) for each CV."""
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    """The NumPy array representing specific heat capacities (J/kg/K) for each CV."""
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    """The NumPy array representing thermal conductivities (W/m/K) for each CV."""
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    """The NumPy array of strings labeling the type of each CV. Not implemented for this deprecated class."""
    raise NotImplementedError("cv_type is not implemented for the deprecated Building class.")

  def reset(self):
    """Resets the building's temperatures and heat inputs to initial states.
    
    Sets all control volume temperatures to `_initial_temp` and all heat inputs
    (`input_q`) to zero.
    """
    # This check is needed because __init__ might not set up dimensions if deprecation=True
    if hasattr(self, '_density') and self._density.size > 0: # Check if arrays were initialized
        matrix_shape = self._density.shape
        self.temp = np.full(matrix_shape, self._initial_temp)
        self.input_q = np.full(matrix_shape, 0.0)
    else: # Fallback for incomplete initialization due to deprecation flag
        self.temp = np.array([])
        self.input_q = np.array([])


  def _calculate_neighbors(
      self, shape: Shape2D
  ) -> List[List[List[Coordinates2D]]]:
    """Calculates the Moore neighborhood (up, down, left, right) for each CV.

    Args:
      shape: A 2-tuple `(rows, cols)` of the building grid.

    Returns:
      A 3D list structure where `neighbors[row][col]` contains a list of
      `(neighbor_row, neighbor_col)` tuples for the CV at `(row, col)`.
    """
    neighbors: List[List[List[Coordinates2D]]] = [[[] for _ in range(shape[1])] for _ in range(shape[0])]
    for r in range(shape[0]):
      for c in range(shape[1]):
        possible_neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for nr, nc in possible_neighbors:
          if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            neighbors[r][c].append((nr, nc))
    return neighbors

  def get_zone_thermal_energy_rate(
      self, zone_coordinates: Coordinates2D
  ) -> float:
    """Calculates total thermal power input (Watts) to a specified zone.

    This sums the `input_q` values for all air CVs within the zone defined by
    `zone_coordinates` in the deprecated grid-based `Building` model.

    Args:
      zone_coordinates: A tuple `(zone_row_idx, zone_col_idx)` identifying the
        zone in the building's room grid.

    Returns:
      The total thermal power input to the zone in Watts (float).
    """
    x_min, x_max, y_min, y_max = get_zone_bounds(zone_coordinates, self.room_shape)
    # Ensure bounds are valid for the current temp array if it was initialized
    if x_min > x_max or y_min > y_max or self.input_q.shape[0] <= x_max or self.input_q.shape[1] <= y_max :
        return 0.0
    submat = self.input_q[x_min : x_max + 1, y_min : y_max + 1]
    return np.sum(submat)

  def get_zone_temp_stats(
      self, zone_coordinates: Coordinates2D
  ) -> Tuple[float, float, float]:
    """Calculates min, max, and mean temperatures for air CVs in a zone.

    Applies to the deprecated grid-based `Building` model.

    Args:
      zone_coordinates: Tuple `(zone_row_idx, zone_col_idx)` identifying the zone.

    Returns:
      A tuple `(min_temp, max_temp, mean_temp)` in Kelvin for the zone.
      Returns `(0.0, 0.0, 0.0)` if bounds are invalid or temp array is empty.
    """
    x_min, x_max, y_min, y_max = get_zone_bounds(zone_coordinates, self.room_shape)
    # Ensure bounds are valid for the current temp array
    if x_min > x_max or y_min > y_max or self.temp.shape[0] <= x_max or self.temp.shape[1] <= y_max:
        return (0.0, 0.0, 0.0)
    submat = self.temp[x_min : x_max + 1, y_min : y_max + 1]
    if submat.size == 0:
        return (0.0, 0.0, 0.0)
    return np.min(submat), np.max(submat), np.mean(submat)

  def get_zone_average_temps(self) -> Dict[Tuple[int, int], float]: # Type hint for value as float
    """Returns a dictionary of average temperatures for each zone.

    Applies to the deprecated grid-based `Building` model. The dictionary maps
    zone coordinates `(zone_row_idx, zone_col_idx)` to average temperatures (K).

    Returns:
      A dictionary `{zone_coordinates: avg_temp_kelvin}`.
    """
    avg_temps: Dict[Tuple[int, int], float] = {}
    # Check if building_shape was initialized (relevant for deprecated class)
    if not hasattr(self, 'building_shape') or not self.building_shape:
        return avg_temps
        
    for zone_x in range(self.building_shape[0]):
      for zone_y in range(self.building_shape[1]):
        zone_coordinates = (zone_x, zone_y)
        _, _, avg_temp = self.get_zone_temp_stats(zone_coordinates)
        avg_temps[zone_coordinates] = avg_temp
    return avg_temps

  def apply_thermal_power_zone(
      self, zone_coordinates: Coordinates2D, power: float
  ) -> None:
    """Applies thermal power (Watts) to a zone, distributed by diffusers.

    Applies to the deprecated grid-based `Building` model. The `power` is
    distributed among the diffuser locations within the specified zone.

    Args:
       zone_coordinates: Tuple `(zone_row_idx, zone_col_idx)` identifying the zone.
       power: Total thermal power in Watts to apply to the zone.
    """
    x_min, x_max, y_min, y_max = get_zone_bounds(zone_coordinates, self.room_shape)
    # Ensure bounds are valid and diffusers array is initialized
    if not (hasattr(self, 'diffusers') and self.diffusers.size > 0 and \
            x_min <= x_max and y_min <= y_max and \
            self.diffusers.shape[0] > x_max and self.diffusers.shape[1] > y_max):
        return

    for x in range(x_min, x_max + 1):
      for y in range(y_min, y_max + 1):
        if self.diffusers[x, y] > 0.0: # If it's a diffuser location
          self.input_q[x, y] = power * self.diffusers[x, y]


@gin.configurable
class FloorPlanBasedBuilding(BaseSimulatorBuilding):
  """Represents a building's thermal structure based on a detailed floor plan.

  This class allows for complex, non-rectangular room geometries by deriving its
  internal representation (walls, air spaces, room definitions) from a
  floor plan image and a zone map. It manages arrays for thermal properties
  (conductivity, heat capacity, density), diffuser layouts, Control Volume (CV)
  types, CV neighbors, and the current temperature and heat input states.

  It provides methods to interact with this representation, such as resetting
  the state, calculating zone temperatures, and applying thermal power to zones.
  It can also use a convection simulator to model air movement effects.

  Attributes:
    cv_size_cm: Size (width, length, height) of each CV in centimeters.
    floor_height_cm: Floor-to-ceiling height of rooms in centimeters.
    floor_plan: NumPy array representing the raw input floor plan.
    temp: NumPy array of current temperatures (K) for each CV.
    input_q: NumPy array of heat energy rates (Watts) applied to each CV.
    diffusers: NumPy array indicating diffuser locations and strengths.
    neighbors: Matrix (list of lists of lists) of neighbor coordinates for each CV.
    len_neighbors: NumPy array storing the number of neighbors for each CV.
    _conductivity: NumPy array of thermal conductivities (W/m/K) for each CV.
    _heat_capacity: NumPy array of specific heat capacities (J/kg/K) for each CV.
    _density: NumPy array of material densities (kg/m^3) for each CV.
    _cv_type: NumPy array of strings labeling the type of each CV.
    _room_dict: Dictionary mapping room names to lists of their CV coordinates.
    _exterior_walls: Binary mask of exterior wall CVs.
    _interior_walls: Binary mask of interior wall CVs.
    _exterior_space: Binary mask of exterior space (outside air) CVs.
    _zone_map: NumPy array representing the VAV zone layout.
    _convection_simulator: Optional convection simulator instance.
    _initial_temp: Initial temperature for resetting the building.
    _reset_temp_values: Optional NumPy array of specific temperatures for reset.
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
      reset_temp_values: Optional[np.ndarray] = None, # Corrected type hint
  ):
    """Initializes the FloorPlanBasedBuilding.

    This involves reading floor plan and zone map data (either from provided
    arrays or filepaths), processing this data to define room boundaries,
    walls (interior and exterior), and air spaces. It then initializes matrices
    for thermal properties, diffuser locations, CV types, and neighbor information.

    Args:
      cv_size_cm: The size (width, length, and height assumed equal) of each
        cubic control volume in centimeters.
      floor_height_cm: The height from floor to ceiling for each room in
        centimeters.
      initial_temp: The initial temperature (in Kelvin) to assign to all
        control volumes if `reset_temp_values` is not provided.
      inside_air_properties: `MaterialProperties` for interior air CVs.
      inside_wall_properties: `MaterialProperties` for interior wall CVs.
      building_exterior_properties: `MaterialProperties` for exterior wall CVs.
      zone_map: A NumPy array where different integer values delineate different
        VAV zones. Mutually exclusive with `zone_map_filepath`.
      zone_map_filepath: Filepath to load the `zone_map` from a .npy file.
        Mutually exclusive with `zone_map`.
      floor_plan: A NumPy array representing the building's layout, where
        different values might indicate walls, open spaces, etc. Mutually
        exclusive with `floor_plan_filepath`.
      floor_plan_filepath: Filepath to load the `floor_plan` from a .npy file.
        Mutually exclusive with `floor_plan`.
      buffer_from_walls: Minimum distance (in CV units) to maintain between
        thermal diffusers and room walls/boundaries.
      convection_simulator: An optional instance of `BaseConvectionSimulator`
        to model air movement effects on temperature.
      reset_temp_values: An optional NumPy array specifying the exact
        temperatures to use for each CV when `reset()` is called. If `None`,
        `initial_temp` is used uniformly.

    Raises:
      ValueError: If both `floor_plan` and `floor_plan_filepath` are None, or
        if both `zone_map` and `zone_map_filepath` are None, or if both are
        provided for either.
    """
    self.cv_size_cm = cv_size_cm
    self.floor_height_cm = floor_height_cm
    self._initial_temp = initial_temp
    self._convection_simulator = convection_simulator
    self._reset_temp_values = reset_temp_values

    # --- Load Floor Plan ---
    if floor_plan is None and floor_plan_filepath is None:
      raise ValueError("Either floor_plan or floor_plan_filepath must be provided.")
    if floor_plan is not None and floor_plan_filepath is not None:
      raise ValueError("Provide either floor_plan or floor_plan_filepath, not both.")
    self.floor_plan = (
        building_utils.read_floor_plan_from_filepath(floor_plan_filepath)
        if floor_plan is None else floor_plan
    )

    # --- Load Zone Map ---
    if zone_map is None and zone_map_filepath is None:
      raise ValueError("Either zone_map or zone_map_filepath must be provided.")
    if zone_map is not None and zone_map_filepath is not None:
      raise ValueError("Provide either zone_map or zone_map_filepath, not both.")
    self._zone_map = (
        building_utils.read_floor_plan_from_filepath(zone_map_filepath)
        if zone_map is None else zone_map
    )

    # --- Process Floor Plan and Zone Map to Define Building Structure ---
    (self._room_dict, exterior_walls_raw, interior_walls_raw, self._exterior_space) = (
        building_utils.construct_building_data_types(
            floor_plan=self.floor_plan, zone_map=self._zone_map
        )
    )
    self._exterior_walls, self._interior_walls = enlarge_exterior_walls(
        exterior_walls=exterior_walls_raw, interior_walls=interior_walls_raw
    )

    # --- Initialize Thermal Property Arrays ---
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

    # --- Initialize Diffusers, CV Types, and Neighbors ---
    self.diffusers = np.zeros(self._exterior_walls.shape)
    self.diffusers = _assign_thermal_diffusers(
        self.diffusers, self._room_dict, self._interior_walls, buffer_from_walls
    )
    self._cv_type = _construct_cv_type_array(self._exterior_walls, self._exterior_space)
    self.neighbors = self._calculate_neighbors()
    self.len_neighbors = self._calculate_length_of_neighbors()

    self.reset() # Initialize temperature and heat input arrays

  @property
  def density(self) -> np.ndarray:
    """The NumPy array representing material densities (kg/m^3) for each CV."""
    return self._density

  @property
  def heat_capacity(self) -> np.ndarray:
    """The NumPy array representing specific heat capacities (J/kg/K) for each CV."""
    return self._heat_capacity

  @property
  def conductivity(self) -> np.ndarray:
    """The NumPy array representing thermal conductivities (W/m/K) for each CV."""
    return self._conductivity

  @property
  def cv_type(self) -> np.ndarray:
    """The NumPy array of strings labeling the type of each CV (e.g., air, wall, exterior)."""
    return self._cv_type

  def reset(self) -> None:
    """Resets the building's thermal state.

    Initializes `self.temp` either with `_reset_temp_values` (if provided) or
    uniformly with `_initial_temp`. Sets `self.input_q` (heat inputs) to zero.
    """
    if self._reset_temp_values is not None:
      self.temp = np.copy(self._reset_temp_values)
    else:
      self.temp = np.full(shape=self._exterior_walls.shape, fill_value=self._initial_temp)
    self.input_q = np.zeros(self._exterior_walls.shape)

  def _calculate_neighbors(self) -> List[List[List[Coordinates2D]]]:
    """Calculates Moore neighbors for all non-exterior-space CVs.

    For each CV in the grid, it identifies its valid neighbors (up, down, left,
    right) that are not part of the exterior space.

    Returns:
      A 3D list structure where `neighbors[row][col]` contains a list of
      `(neighbor_row, neighbor_col)` tuples for the CV at `(row, col)`.
      CVs corresponding to exterior space will have an empty list of neighbors.
    """
    shape = self._exterior_walls.shape
    # Initialize a grid of empty lists for neighbors
    neighbors: List[List[List[Coordinates2D]]] = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    for r in range(shape[0]):
      for c in range(shape[1]):
        if self.cv_type[r, c] == constants.LABEL_FOR_EXTERIOR_SPACE:
          continue # Exterior space CVs have no internal neighbors for thermal sim

        possible_neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for nr, nc in possible_neighbors:
          # Check bounds
          if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            # Add if neighbor is not also exterior space
            if self.cv_type[nr, nc] != constants.LABEL_FOR_EXTERIOR_SPACE:
              neighbors[r][c].append((nr, nc))
    return neighbors

  def _calculate_length_of_neighbors(self) -> np.ndarray:
    """Creates an array storing the number of neighbors for each CV.

    Returns:
      A NumPy array of the same shape as the building grid, where each cell
      value is the count of valid neighbors for that CV.
    """
    len_neighbors_arr = np.zeros(shape=self._exterior_walls.shape, dtype=int)
    for r in range(self.neighbors_shape[0]): # Assuming neighbors_shape is available
      for c in range(self.neighbors_shape[1]):
        len_neighbors_arr[r, c] = len(self.neighbors[r][c])
    return len_neighbors_arr
  
  @property # Helper property for cleaner access in _calculate_length_of_neighbors
  def neighbors_shape(self) -> Shape2D:
      """Returns the shape of the neighbors grid (and other main grid arrays)."""
      if isinstance(self.neighbors, np.ndarray): # Should be list of lists
          return self.neighbors.shape[:2]
      elif isinstance(self.neighbors, list) and self.neighbors and isinstance(self.neighbors[0], list):
          return (len(self.neighbors), len(self.neighbors[0]))
      return (0,0) # Fallback

  def get_zone_thermal_energy_rate(self, zone_name: str) -> float:
    """Calculates total thermal power input (Watts) to a specified zone.

    Sums the `input_q` values for all CVs belonging to the given `zone_name`
    as defined in `self._room_dict`.

    Args:
      zone_name: The string identifier of the zone (must be a key in
        `self._room_dict`).

    Returns:
      The total thermal power input to the zone in Watts (float).

    Raises:
      ValueError: If `zone_name` is not found in `self._room_dict`.
    """
    if zone_name not in self._room_dict:
      raise ValueError(f"Zone '{zone_name}' not found in room_dict.")

    zone_cv_coords = self._room_dict[zone_name]
    # Sum input_q values for the CVs in the specified zone
    total_q_for_zone = sum(self.input_q[coord] for coord in zone_cv_coords)
    return float(np.sum(total_q_for_zone)) # Ensure float return

  def get_zone_temp_stats(self, zone_name: str) -> Tuple[float, float, float]:
    """Calculates min, max, and mean temperatures for a specified zone.

    Args:
      zone_name: The string identifier of the zone (must be a key in
        `self._room_dict`).

    Returns:
      A tuple `(min_temp, max_temp, mean_temp)` in Kelvin for the zone's CVs.
      Returns `(nan, nan, nan)` if the zone has no CVs or is not found.

    Raises:
      ValueError: If `zone_name` is not found in `self._room_dict`.
    """
    if zone_name not in self._room_dict:
      raise ValueError(f"Zone '{zone_name}' not found in room_dict.")

    zone_cv_coords = self._room_dict[zone_name]
    if not zone_cv_coords: # Check if the list of coordinates is empty
        return (np.nan, np.nan, np.nan)
        
    # Extract temperatures for the CVs in the specified zone
    zone_temps = [self.temp[coord] for coord in zone_cv_coords]
    if not zone_temps: # Double check if temps list ended up empty
        return (np.nan, np.nan, np.nan)
        
    return (float(np.min(zone_temps)), float(np.max(zone_temps)), float(np.mean(zone_temps)))

  def get_zone_average_temps(self) -> Dict[str, float]: # Value type is float
    """Returns a dictionary of average temperatures for each defined room/zone.

    The dictionary maps room/zone names (strings, from `self._room_dict` keys
    that start with `constants.ROOM_STRING_DESIGNATOR`) to their average
    temperatures in Kelvin.

    Returns:
      A dictionary `{zone_name: avg_temp_kelvin}`.
    """
    avg_temps: Dict[str, float] = {}
    for zone_key in self._room_dict:
      # Ensure only actual rooms/zones are processed, not other entries
      if zone_key.startswith(constants.ROOM_STRING_DESIGNATOR):
        _, _, avg_temp = self.get_zone_temp_stats(zone_key)
        if not np.isnan(avg_temp): # Only add if stats were valid
            avg_temps[zone_key] = avg_temp
    return avg_temps

  def apply_thermal_power_zone(self, zone_name: str, power: float) -> None:
    """Applies thermal power (Watts) to a zone, distributed by its diffusers.

    The total `power` is distributed among the diffuser locations within the
    specified `zone_name`. The proportion of power to each diffuser CV is
    determined by the values in `self.diffusers`.

    Args:
      zone_name: The string identifier of the zone to apply power to (must be
        a key in `self._room_dict`).
      power: Total thermal power in Watts to apply to the zone. Positive for
        heating, negative for cooling.

    Raises:
      ValueError: If `zone_name` is not found in `self._room_dict`.
    """
    if zone_name not in self._room_dict:
      raise ValueError(f"Zone '{zone_name}' not found in room_dict.")

    zone_cv_coords = self._room_dict[zone_name]
    for coord_tuple in zone_cv_coords:
      # Ensure coord_tuple is correctly indexed if it's from a structured array or different format
      coord = tuple(coord_tuple) # Ensure it's a simple tuple for indexing
      if self.diffusers[coord] > 0.0: # If it's a diffuser location
        self.input_q[coord] = power * self.diffusers[coord]

  def apply_convection(self) -> None:
    """Applies the convection simulation step if a simulator is configured.
    
    If `self._convection_simulator` is set, its `apply_convection` method
    is called with the current building's `_room_dict` and `temp` array.
    This modifies `self.temp` in place to simulate heat transfer due to
    air movement.
    """
    if self._convection_simulator is not None:
      self._convection_simulator.apply_convection(self._room_dict, self.temp)
