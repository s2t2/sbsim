"""Utilities for processing building floor plans for thermal simulation.

This module provides a suite of functions for reading, parsing, and transforming
building floor plan data into representations suitable for initializing thermal
simulation models like `FloorPlanBasedBuilding`. Key functionalities include:
- Reading floor plan data from files (.csv, .npy).
- Ensuring floor plans have adequate exterior padding.
- Identifying connected components (rooms/zones) using OpenCV.
- Labeling different types of spaces: exterior space, interior space,
  exterior walls, and interior walls.
- Constructing a dictionary mapping room/zone names to their constituent
  control volume (CV) coordinates.
- Helper functions for image-based debugging and geometric operations like
  enlarging components.

The module defines several `NewType` aliases (e.g., `FileInputFloorPlan`,
`Connections`) to clarify the state of the floor plan data as it undergoes
various processing stages. These types are primarily for semantic clarity and
are based on NumPy arrays with specific integer encodings representing different
physical elements (walls, air, etc.).
"""

import collections
import datetime
import pathlib
from typing import Any, Dict, List, NewType, Sequence, Tuple, Union # Added Dict, Sequence
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage # For ndimage.generate_binary_structure

from smart_control.simulator import constants

# Type alias for 2D coordinates (row, column)
Coordinates2D = Tuple[int, int]

# Type alias for a dictionary mapping room names (str) to a sequence of their
# constituent CV coordinates. Uses collections.defaultdict for convenience if needed.
RoomIndicesDict = Dict[str, Sequence[Coordinates2D]] # Changed from defaultdict for common usage

# --- Custom Types for Floor Plan Processing Stages ---
# These NewType aliases help in clarifying the expected format and interpretation
# of NumPy arrays at different points in the floor plan processing pipeline.

# FileInputFloorPlan: Represents the raw floor plan data as loaded from a file or
#   provided as input. Typically uses specific integer encodings:
#   - `constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT` (e.g., 2) for exterior space.
#   - `constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT` (e.g., 1) for walls.
#   - `constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT` (e.g., 0) for interior air/space.
#   This input might have inconsistencies like walls at the very edge of the array,
#   which subsequent functions aim to correct.
FileInputFloorPlan = NewType("FileInputFloorPlan", np.ndarray)

# ConnectionReadyFloorPlan: Represents a floor plan processed to be suitable for
#   input into OpenCV's `connectedComponentsWithStats` function. This format
#   requires a binary representation where:
#   - `constants.INTERIOR_SPACE_VALUE_IN_CONNECTION_INPUT` (e.g., 1) marks
#     interior spaces (the components to be identified).
#   - `constants.GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT` (e.g., 0) marks
#     everything else (walls, exterior space).
ConnectionReadyFloorPlan = NewType("ConnectionReadyFloorPlan", np.ndarray)

# Connections: Represents the output of `connectedComponentsWithStats`.
#   It's a NumPy array where each distinct connected component (room/zone)
#   is labeled with a unique positive integer. Walls and unclassified areas
#   are typically labeled 0 by the OpenCV function. Exterior space might be
#   post-processed to a specific negative value.
Connections = NewType("Connections", np.ndarray)

# ExteriorSpace: A binary mask identifying exterior space.
#   - `constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION` (e.g., -1 or a specific positive value
#     if processed differently) marks exterior CVs.
#   - Other values (e.g., 0) mark interior spaces or walls.
ExteriorSpace = NewType("ExteriorSpace", np.ndarray)

# ExteriorWalls: A binary mask identifying exterior wall locations.
#   - `constants.EXTERIOR_WALL_VALUE_IN_FUNCTION` (e.g., 1) marks exterior wall CVs.
#   - 0 marks other CVs.
ExteriorWalls = NewType("ExteriorWalls", np.ndarray)

# InteriorWalls: A binary mask identifying interior wall locations.
#   - `constants.INTERIOR_WALL_VALUE_IN_FUNCTION` (e.g., -3 or a specific positive value)
#     marks interior wall CVs.
#   - 0 marks other CVs.
InteriorWalls = NewType("InteriorWalls", np.ndarray)
# --- End of Custom Types ---


def read_floor_plan_from_filepath(
    filepath: str,
    save_debugging_image: bool = False,
) -> FileInputFloorPlan:
  """Reads floor plan data from a .csv or .npy file.

  Args:
    filepath: The absolute or relative path to the floor plan file.
      The file extension (`.csv` or `.npy`) determines how it's parsed.
    save_debugging_image: If True, saves a visualization of the loaded
      floor plan as a PNG image for debugging purposes. The image is saved
      to a predefined CNS path with a timestamped filename.

  Returns:
    A `FileInputFloorPlan` (NumPy array) representing the loaded floor plan.

  Raises:
    ValueError: If the file extension is not `.csv` or `.npy`.
    FileNotFoundError: If the specified `filepath` does not exist.
  """
  file_path_obj = pathlib.Path(filepath)
  file_suffix = file_path_obj.suffix

  if not file_path_obj.exists():
    raise FileNotFoundError(f"Floor plan file not found at: {filepath}")

  with file_path_obj.open(mode="rb") as fp:
    if file_suffix == ".csv":
      floor_plan_data = np.loadtxt(fp, delimiter=",")
    elif file_suffix == ".npy":
      floor_plan_data = np.load(fp, allow_pickle=True) # allow_pickle for older .npy files
    else:
      raise ValueError(
          f"Unsupported file type '{file_suffix}'. Please provide a .csv or .npy file."
      )

  floor_plan_array = np.asarray(floor_plan_data, dtype=np.int16) # Ensure consistent dtype

  if save_debugging_image:
    # Assuming save_images_to_cns_for_debugging handles potential CNS path issues
    save_images_to_cns_for_debugging(
        FileInputFloorPlan(floor_plan_array), "file_from_input"
    )
  return FileInputFloorPlan(floor_plan_array)


def save_images_to_cns_for_debugging(
    floor_plan_array: Union[ # More specific union of types
        FileInputFloorPlan, ConnectionReadyFloorPlan, Connections,
        ExteriorSpace, ExteriorWalls, InteriorWalls, np.ndarray
    ],
    path_ending: str,
    # Default path is illustrative; actual CNS paths may vary.
    path_to_simulator_cns: str = "/cns/oi-d/home/smart_buildings/control/configs/simulation/",
) -> None:
  """Saves a NumPy array representing a floor plan stage as a PNG image.

  This utility is for visual debugging of the floor plan processing steps.
  It saves the input array as an image to a specified CNS (Google Cloud Storage)
  path, appending a descriptive `path_ending` and the current date to the
  filename.

  Args:
    floor_plan_array: The NumPy array to be visualized and saved. This can be
      any of the custom floor plan types or a generic NumPy array.
    path_ending: A string suffix to append to the image filename for
      identification (e.g., "initial_load", "after_padding").
    path_to_simulator_cns: The base CNS directory path where the debug image
      will be saved. A subdirectory "floorplan_construction_debugging_images"
      will be used within this path.
  """
  # Construct a unique filename with a date stamp
  filename = f"{path_ending}_{datetime.datetime.now().strftime('%Y%m%d')}.png"
  # Ensure the target directory exists. For CNS, this might not be needed if
  # the path library handles it, but good practice for local saving.
  # For CNS, direct matplotlib saving might not work; consider saving locally
  # then uploading, or using a CNS-compatible file writing method.
  # This implementation assumes matplotlib can write to a pathlib.Path object
  # that might represent a CNS path if configured correctly with tf.io.gfile.
  # However, standard pathlib does not directly handle CNS.
  # For simplicity here, we'll assume local path or properly configured TF gfile.

  # Create a pathlib.Path object. If path_to_simulator_cns is a local path,
  # this will work directly. For CNS, tf.io.gfile would be needed.
  # This example will proceed as if it's a local path for matplotlib.
  # A more robust solution for CNS would use tf.io.gfile.
  try:
    # Define a local temporary path if CNS direct saving is an issue
    local_temp_dir = pathlib.Path("/tmp/floorplan_debug_images")
    local_temp_dir.mkdir(parents=True, exist_ok=True)
    local_save_path = local_temp_dir / filename
    
    plt.imshow(floor_plan_array) # type: ignore # imshow handles array-like
    plt.title(path_ending) # Add title for context
    plt.colorbar() # Add colorbar to understand values
    plt.savefig(local_save_path)
    plt.close() # Close the figure to free memory
    print(f"Debug image saved locally to: {local_save_path}")
    # If CNS is the target, add code here to upload local_save_path to
    # target_cns_path = pathlib.Path(path_to_simulator_cns) / "floorplan_construction_debugging_images" / filename
    # e.g., using tf.io.gfile.copy(str(local_save_path), str(target_cns_path), overwrite=True)
  except Exception as e: # pylint: disable=broad-except
    warnings.warn(f"Could not save debug image {filename}: {e}")


def guarantee_air_padding_in_frame(
    floor_plan: FileInputFloorPlan,
) -> FileInputFloorPlan:
  """Ensures the floor plan is surrounded by at least one layer of exterior space.

  Building simulation algorithms often rely on boundary conditions that assume
  the building is not directly abutting the edge of the simulation grid. This
  function inspects the edges of the `floor_plan` array. If any wall CVs
  (typically encoded as 1) are found at the outermost rows or columns, it adds
  a new row/column of exterior space CVs (encoded as
  `constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT`) to that side. This process
  is repeated for all four edges.

  Args:
    floor_plan: A `FileInputFloorPlan` (NumPy array) representing the raw
      building layout.

  Returns:
    The `FileInputFloorPlan` (NumPy array), potentially with added padding of
    exterior space CVs around its borders.

  Raises:
    ValueError: If the input `floor_plan` has a dimension of size 0 or 1,
      making padding ambiguous or problematic.
  """
  if 0 in floor_plan.shape or 1 in floor_plan.shape:
    raise ValueError(
        "Floor plan has a trivial dimension (0 or 1), cannot apply padding."
    )

  current_floor_plan = floor_plan.copy() # Work on a copy

  # Helper to create a row/column of exterior space matching the current dimensions
  def _get_padding_row(num_cols: int) -> np.ndarray:
    return np.full((1, num_cols), constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT, dtype=current_floor_plan.dtype)
  def _get_padding_col(num_rows: int) -> np.ndarray:
    return np.full((num_rows, 1), constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT, dtype=current_floor_plan.dtype)

  # Check and pad top edge
  if np.any(current_floor_plan[0, :] == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT):
    current_floor_plan = np.concatenate((_get_padding_row(current_floor_plan.shape[1]), current_floor_plan), axis=0)
  # Check and pad left edge
  if np.any(current_floor_plan[:, 0] == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT):
    current_floor_plan = np.concatenate((_get_padding_col(current_floor_plan.shape[0]), current_floor_plan), axis=1)
  # Check and pad bottom edge
  if np.any(current_floor_plan[-1, :] == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT):
    current_floor_plan = np.concatenate((current_floor_plan, _get_padding_row(current_floor_plan.shape[1])), axis=0)
  # Check and pad right edge
  if np.any(current_floor_plan[:, -1] == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT):
    current_floor_plan = np.concatenate((current_floor_plan, _get_padding_col(current_floor_plan.shape[0])), axis=1)

  return FileInputFloorPlan(current_floor_plan)


def _determine_exterior_space(
    floor_plan: FileInputFloorPlan,
) -> Tuple[ConnectionReadyFloorPlan, ExteriorSpace]:
  """Separates exterior space and prepares interior space for connected components.

  This function takes a `FileInputFloorPlan` and performs two transformations:
  1.  Creates an `ExteriorSpace` mask: Marks locations corresponding to
      `constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT` with a functional constant
      (`constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION`) and all other locations
      (interior space, walls) with a generic non-component value.
  2.  Creates a `ConnectionReadyFloorPlan`: Marks locations corresponding to
      `constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT` as connectable components
      (e.g., value 1) and all other locations (exterior space, walls) as
      non-connectable background (e.g., value 0). This format is required by
      OpenCV's `connectedComponentsWithStats`.

  Args:
    floor_plan: A `FileInputFloorPlan` NumPy array.

  Returns:
    A tuple `(connection_ready_plan, exterior_space_mask)`:
      - `connection_ready_plan`: The `ConnectionReadyFloorPlan` NumPy array.
      - `exterior_space_mask`: The `ExteriorSpace` NumPy array.
  """
  # Create the exterior space mask
  exterior_space_mask = np.where(
      floor_plan == constants.EXTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,  # Mark exterior space
      constants.GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT,  # Mark others as non-component/background
  )
  # Prepare floor plan for connected components: interior spaces become 1, others 0
  connection_ready_plan = np.where(
      floor_plan == constants.INTERIOR_SPACE_VALUE_IN_FILE_INPUT,
      constants.INTERIOR_SPACE_VALUE_IN_CONNECTION_INPUT, # Mark interior as connectable
      constants.GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT,  # Mark walls/exterior as background
  )
  return ConnectionReadyFloorPlan(connection_ready_plan), ExteriorSpace(exterior_space_mask)


def _run_connected_components(
    floor_plan: ConnectionReadyFloorPlan,
    connectivity: int = 4,
    save_debugging_image: bool = False,
) -> Connections:
  """Identifies and labels connected components (rooms/zones) in a floor plan.

  Uses OpenCV's `cv2.connectedComponentsWithStats` function to find distinct
  contiguous regions of interior space in the `ConnectionReadyFloorPlan`.
  Each identified component is assigned a unique positive integer label.
  Background (walls, exterior) is typically labeled 0 by the function.

  Args:
    floor_plan: A `ConnectionReadyFloorPlan` (binary NumPy array) where
      interior spaces are marked as components to be connected.
    connectivity: Defines the neighborhood for connectivity. Use 4 for
      4-connectivity (only considers horizontal/vertical neighbors) or 8 for
      8-connectivity (also considers diagonal neighbors). Defaults to 4.
    save_debugging_image: If True, saves a visualization of the labeled
      components array for debugging.

  Returns:
    A `Connections` NumPy array where each connected component (room/zone)
    is labeled with a unique integer.
  """
  # Ensure input is uint8 as required by connectedComponentsWithStats
  binary_input_map = np.uint8(floor_plan.copy())

  # num_labels includes the background label (0)
  # labels is the output array with each component assigned a unique integer
  # stats provides bounding box and area for each component (not used here)
  # centroids provides centroids of components (not used here)
  num_labels, labels, _, _ = cv2.connectedComponentsWithStats( # pylint: disable=unused-variable
      binary_input_map, connectivity=connectivity
  )

  # Warn if an unusually small number of components (rooms) are found,
  # which might indicate an issue with the input floor plan encoding.
  # (e.g., if 0s and 1s were inverted). np.max(labels) gives num_labels - 1.
  if num_labels -1 < 5 : # num_labels includes background
    warnings.warn(
        f"Connected components found only {num_labels-1} room(s)/zone(s). "
        "Check if the input floor plan correctly marks interior spaces for "
        "connection (e.g., as 1s) and barriers as 0s."
    )

  if save_debugging_image:
    save_images_to_cns_for_debugging(Connections(labels), "connections_output")

  return Connections(labels)


def _set_exterior_space_neg(
    connections: Connections, exterior_space: ExteriorSpace
) -> Connections:
  """Marks exterior space regions with a negative value in the connections array.

  This function updates the `connections` array (output from
  `_run_connected_components`) by using the `exterior_space` mask.
  Locations identified as exterior space are assigned a specific negative
  constant (`constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION`). This helps to
  semantically distinguish exterior space from interior components (which have
  positive integer labels) and walls (typically label 0).

  Args:
    connections: A `Connections` NumPy array where interior components are
      labeled with positive integers.
    exterior_space: An `ExteriorSpace` NumPy array (binary mask) identifying
      exterior space locations.

  Returns:
    The modified `Connections` NumPy array where exterior space is labeled
    with a negative value.
  """
  # Where exterior_space mask is true (marks exterior), set connections to negative constant.
  # Otherwise, keep the existing connection label.
  updated_connections = np.where(
      exterior_space == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION,
      constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION, # Assign the negative constant
      connections, # Keep original label for interior/walls
  )
  return Connections(updated_connections)


def _label_exterior_wall_shell(
    exterior_space: ExteriorSpace,
) -> ExteriorWalls:
  """Identifies the "shell" of exterior walls bordering exterior space.

  This function creates a binary mask highlighting only the Control Volumes (CVs)
  that represent the innermost layer of exterior walls—those CVs that are
  directly adjacent to `exterior_space` but are not `exterior_space` themselves.
  It uses binary dilation to find all CVs near exterior space, then subtracts
  the exterior space itself to leave just the wall shell.

  Args:
    exterior_space: An `ExteriorSpace` NumPy array (binary mask) where non-zero
      values (e.g., `constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION`) mark
      exterior space.

  Returns:
    An `ExteriorWalls` NumPy array (binary mask) where exterior wall shell
    locations are marked with `constants.EXTERIOR_WALL_VALUE_IN_FUNCTION` (e.g., 1)
    and all other locations are 0.
  """
  # Create a binary mask: True for exterior space, False otherwise.
  is_exterior_space_mask = (exterior_space == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION)

  # Define a structuring element for dilation (4-connectivity).
  struct = ndimage.generate_binary_structure(rank=2, connectivity=1) # rank=2 for 2D, conn=1 for 4-way

  # Dilate the exterior space mask to include immediately adjacent CVs.
  is_near_exterior_space_mask = ndimage.binary_dilation(
      is_exterior_space_mask, structure=struct
  )

  # The exterior wall shell is where `is_near_exterior_space_mask` is True
  # AND `is_exterior_space_mask` is False.
  is_exterior_wall_shell_mask = is_near_exterior_space_mask & ~is_exterior_space_mask

  # Create the output array with the defined constant for exterior walls.
  exterior_wall_shell_array = np.where(
      is_exterior_wall_shell_mask, constants.EXTERIOR_WALL_VALUE_IN_FUNCTION, 0
  ).astype(np.int16) # Ensure consistent type

  return ExteriorWalls(exterior_wall_shell_array)


def _label_interior_walls(
    exterior_walls: ExteriorWalls, # This is actually the exterior wall *shell* from previous step
    original_floor_plan: FileInputFloorPlan,
) -> InteriorWalls:
  """Identifies interior wall locations from the original floor plan.

  This function creates a binary mask for interior walls. It starts by assuming
  all non-exterior-wall-shell locations could potentially be interior space or
  interior walls. Then, it specifically marks locations as interior walls if they
  were defined as such (`constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT`) in the
  `original_floor_plan`. Finally, it ensures that any locations already
  identified as part of the `exterior_walls` shell are *not* marked as interior
  walls.

  Args:
    exterior_walls: An `ExteriorWalls` NumPy array (binary mask) representing
      the shell of exterior walls.
    original_floor_plan: The `FileInputFloorPlan` NumPy array, used to identify
      the original locations of interior walls.

  Returns:
    An `InteriorWalls` NumPy array (binary mask) where interior wall locations
    are marked with `constants.INTERIOR_WALL_VALUE_IN_FUNCTION` and all other
    locations are 0.
  """
  # Start with an array assuming all non-exterior-shell is interior space (or interior wall)
  # This means locations that are exterior_wall_shell get 0, others get potential interior wall value.
  # This logic seems a bit complex. A clearer way might be:
  # 1. Create a mask of where original_floor_plan == INTERIOR_WALL_VALUE_IN_FILE_INPUT
  # 2. Ensure these locations are not also part of exterior_walls.
  
  # Current logic:
  # Initialize with all INTERIOR_SPACE_VALUE_IN_FUNCTION (e.g., 0)
  interior_walls_mask = np.full_like(exterior_walls, constants.INTERIOR_SPACE_VALUE_IN_FUNCTION)
  
  # Mark locations that were originally interior walls
  interior_walls_mask[
      original_floor_plan == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT
  ] = constants.INTERIOR_WALL_VALUE_IN_FUNCTION # Mark as interior wall
  
  # Ensure that exterior walls are not marked as interior walls
  interior_walls_mask[
      exterior_walls == constants.EXTERIOR_WALL_VALUE_IN_FUNCTION
  ] = constants.INTERIOR_SPACE_VALUE_IN_FUNCTION # Reset to non-interior-wall if it's an exterior wall

  return InteriorWalls(interior_walls_mask.astype(np.int16))


def _construct_room_dict(connections: Connections) -> RoomIndicesDict:
  """Creates a dictionary mapping room/zone names to their CV coordinates.

  This function processes a `Connections` array (where each distinct room/zone
  is labeled with a unique integer, and exterior space has a specific negative
  label). It iterates through the `connections` array and groups all
  `(row, col)` coordinates by their component label.

  Args:
    connections: A `Connections` NumPy array where connected components
      (rooms/zones) are labeled with positive integers, and exterior space is
      labeled with `constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION`. Walls are
      typically labeled 0.

  Returns:
    A `RoomIndicesDict` (dictionary) where keys are string identifiers
    (e.g., "room_1", "room_2", `constants.EXTERIOR_SPACE_NAME_IN_ROOM_DICT`)
    and values are lists of `(row, col)` tuples representing the CV coordinates
    belonging to that room/zone or space.
  """
  room_dict: RoomIndicesDict = collections.defaultdict(list)

  # Helper to generate descriptive names for components
  def _component_to_room_name(component_label: int) -> str:
    if component_label == constants.EXTERIOR_SPACE_VALUE_IN_FUNCTION:
      return constants.EXTERIOR_SPACE_NAME_IN_ROOM_DICT
    elif component_label == 0: # Typically background/walls from connectedComponents
      # Decide if walls should be in room_dict. Original code implies they might be.
      # If `constants.INTERIOR_WALL_VALUE_IN_COMPONENT` is 0, this handles it.
      return constants.INTERIOR_WALL_NAME_IN_ROOM_DICT # Or some other generic wall name
    else: # Positive integers are room/zone labels
      return f"{constants.ROOM_STRING_DESIGNATOR}{component_label}"

  # Iterate through the connections grid
  for r_idx in range(connections.shape[0]):
    for c_idx in range(connections.shape[1]):
      component_label = connections[r_idx, c_idx]
      room_name = _component_to_room_name(component_label)
      room_dict[room_name].append((r_idx, c_idx))
  return room_dict


def process_and_run_connected_components(
    floor_plan: FileInputFloorPlan,
) -> Connections:
  """Processes a raw floor plan to identify and label connected components.

  This is a higher-level utility function that chains several processing steps:
  1.  Prepares the `floor_plan` for connected components analysis by separating
      exterior space (`_determine_exterior_space`).
  2.  Runs OpenCV's connected components algorithm (`_run_connected_components`)
      to label distinct interior regions.
  3.  Marks the identified exterior space with a specific negative label in the
      resulting components array (`_set_exterior_space_neg`).

  Args:
    floor_plan: A `FileInputFloorPlan` NumPy array.

  Returns:
    A `Connections` NumPy array where interior rooms/zones are labeled with
    positive integers, walls are typically 0, and exterior space is marked with
    a negative value.
  """
  connection_ready_plan, exterior_space_mask = _determine_exterior_space(floor_plan)
  # Run connected components on the plan where only interior spaces are marked for connection
  labeled_connections = _run_connected_components(connection_ready_plan, connectivity=4)
  # Update the labeled connections to specifically mark exterior space
  final_connections = _set_exterior_space_neg(labeled_connections, exterior_space_mask)
  return final_connections


def construct_building_data_types(
    floor_plan: FileInputFloorPlan,
    zone_map: FileInputFloorPlan, # Zone map is also a FileInputFloorPlan type
    save_debugging_image: bool = False,
) -> Tuple[RoomIndicesDict, ExteriorWalls, InteriorWalls, ExteriorSpace]:
  """Orchestrates the full processing of floor plan and zone map data.

  This is a key public function that takes a raw `floor_plan` (defining physical
  layout like walls and open spaces) and a `zone_map` (defining VAV zone
  assignments) and converts them into structured data types required for
  initializing a `FloorPlanBasedBuilding` model.

  The process involves:
  1.  Padding both floor plan and zone map to ensure exterior boundaries.
  2.  Identifying exterior space from the padded floor plan.
  3.  Labeling exterior wall shells and interior walls based on the padded floor plan.
  4.  Running connected components on the (padded) zone map to identify distinct zones.
  5.  Constructing a `RoomIndicesDict` that maps zone names to their CV coordinates.

  Args:
    floor_plan: A `FileInputFloorPlan` NumPy array representing the physical
      layout (walls, interior spaces, exterior spaces).
    zone_map: A `FileInputFloorPlan` NumPy array where integer values define
      different VAV zones. The encoding should align with `FileInputFloorPlan`
      for walls and spaces if they are to be excluded from zones.
    save_debugging_image: If True, saves intermediate processing stage images
      (e.g., wall masks) for debugging.

  Returns:
    A tuple `(room_indices_dict, exterior_wall_mask, interior_wall_mask, exterior_space_mask)`:
      - `room_indices_dict`: Maps zone names to lists of their CV coordinates.
      - `exterior_wall_mask`: Binary mask of exterior walls.
      - `interior_wall_mask`: Binary mask of interior walls.
      - `exterior_space_mask`: Binary mask of exterior space.
  """
  # Ensure both floor_plan and zone_map have adequate exterior padding
  padded_floor_plan = guarantee_air_padding_in_frame(floor_plan)
  padded_zone_map = guarantee_air_padding_in_frame(zone_map) # Zone map might also need padding

  # It seems merged_floor_zone is not used later; its purpose might be for a combined view.
  # merged_floor_zone = padded_floor_plan.copy()
  # merged_floor_zone = np.where(padded_zone_map == constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
  #                             constants.INTERIOR_WALL_VALUE_IN_FILE_INPUT,
  #                             merged_floor_zone) # Assuming 1 in zone_map means zone, not wall

  # Determine exterior space and wall structures from the physical floor plan
  _, exterior_space_mask = _determine_exterior_space(padded_floor_plan)
  exterior_wall_mask = _label_exterior_wall_shell(exterior_space_mask)
  interior_wall_mask = _label_interior_walls(exterior_wall_mask, padded_floor_plan)

  if save_debugging_image:
    save_images_to_cns_for_debugging(ExteriorWalls(exterior_wall_mask), "exterior_walls_processed")
    save_images_to_cns_for_debugging(InteriorWalls(interior_wall_mask), "interior_walls_processed")

  # Process the zone_map to identify distinct zones and create the room/zone dictionary
  # The `process_and_run_connected_components` expects a FileInputFloorPlan where
  # interior spaces to be connected are marked appropriately.
  # If zone_map uses 0 for walls/non-zone and positive integers for zones, it might need
  # transformation to fit the `FileInputFloorPlan` expectation if it assumes 0 is interior.
  # Assuming padded_zone_map is already in a format where zones are connectable areas.
  # For example, if zones in zone_map are marked as INTERIOR_SPACE_VALUE_IN_FILE_INPUT (e.g. 0)
  # and barriers between zones as INTERIOR_WALL_VALUE_IN_FILE_INPUT (e.g. 1).
  zone_connections = process_and_run_connected_components(padded_zone_map)
  room_indices_dict = _construct_room_dict(zone_connections)

  return room_indices_dict, exterior_wall_mask, interior_wall_mask, exterior_space_mask


def enlarge_component(
    array_with_component_nonzero: np.ndarray, distance_to_augment: float
) -> np.ndarray:
  """Enlarges a binary component mask by a specified Euclidean distance.

  This function takes a binary NumPy array where non-zero values represent a
  component. It computes the distance transform from the background (zero values)
  and then thresholds this distance map. The result is a new binary mask where
  the original component is "thickened" or "enlarged" to include all pixels
  within `distance_to_augment` of the original component.

  Args:
    array_with_component_nonzero: A binary NumPy array where the component to
      be enlarged is marked with non-zero values.
    distance_to_augment: The Euclidean distance by which to enlarge the
      component. All pixels within this distance of the original component
      will be included in the result.

  Returns:
    A NumPy array of `np.uint8` type, where 1 indicates pixels belonging to the
    enlarged component, and 0 indicates background.
  """
  # Ensure input is binary (0 or 1) and of type uint8 for distanceTransform
  binary_input = np.uint8(array_with_component_nonzero != 0)
  # Invert the image so background is non-zero for distance calculation from background
  inverted_binary_input = 1 - binary_input

  # Compute Euclidean distance from non-zero pixels (background)
  # cv2.DIST_L2 is Euclidean distance.
  # maskSize=3 means a 3x3 mask for distance calculation.
  distances = cv2.distanceTransform(inverted_binary_input, cv2.DIST_L2, maskSize=3)
  # No need to round here as we threshold directly.

  # Create a mask where distance is less than or equal to distance_to_augment
  # This selects pixels that are "close enough" to the original component.
  enlarged_mask = np.uint8(distances <= distance_to_augment)

  return enlarged_mask
