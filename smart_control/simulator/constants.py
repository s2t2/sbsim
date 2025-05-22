"""Defines constants for use in the building thermal simulation code suite.

This module centralizes various fixed values used for:
- Encoding different types of Control Volumes (CVs) such as walls, interior
  air, and exterior space at various stages of floor plan processing.
- String labels for these CV types.
- Parameters for geometric processing of floor plans.
- Default paths or limits used within the simulation.

Using constants helps maintain consistency and makes it easier to understand
and modify these conventional values.
"""

# --- Floor Plan Processing Encodings ---
# These constants define integer values used to represent different elements
# within NumPy arrays that hold floor plan or processed building data.

# INTERIOR_WALL_VALUE_IN_FUNCTION: Value used to mark interior walls during
# internal processing steps, chosen to be distinct from OpenCV's connected
# components labels (which are positive) and initial file input values.
INTERIOR_WALL_VALUE_IN_FUNCTION = -3

# INTERIOR_WALL_VALUE_IN_COMPONENT: Value representing interior walls after
# the connected components algorithm has run. Typically, connectedComponents
# labels components with positive integers and background (walls) as 0.
INTERIOR_WALL_VALUE_IN_COMPONENT = 0

# EXTERIOR_WALL_VALUE_IN_FUNCTION: Value used to mark exterior walls during
# internal processing, distinct from other encodings.
EXTERIOR_WALL_VALUE_IN_FUNCTION = -2

# EXTERIOR_SPACE_VALUE_IN_FILE_INPUT: The integer value expected in raw input
# floor plan files (e.g., CSV, NPY) to designate exterior space/ambient air.
EXTERIOR_SPACE_VALUE_IN_FILE_INPUT = 2

# EXTERIOR_SPACE_VALUE_IN_FUNCTION: Value used to represent exterior space
# during internal processing. A negative value is chosen to keep positive
# integers available for labeling distinct interior rooms/zones by
# connected components analysis.
EXTERIOR_SPACE_VALUE_IN_FUNCTION = -1

# EXTERIOR_SPACE_VALUE_IN_COMPONENT: Value representing exterior space in the
# array that has been processed by connected components. Often, exterior space
# might be set to 0 or a specific negative value in this array.
# The original value '0' might conflict if walls are also 0. This might need context.
EXTERIOR_SPACE_VALUE_IN_COMPONENT = 0 # Note: This might be ambiguous if walls are also 0.

# INTERIOR_SPACE_VALUE_IN_FILE_INPUT: The integer value expected in raw input
# floor plan files to designate interior air/space within rooms.
INTERIOR_SPACE_VALUE_IN_FILE_INPUT = 0

# INTERIOR_SPACE_VALUE_IN_FUNCTION: Value used to represent interior (room) air
# space during internal processing steps, particularly after distinguishing it
# from walls and exterior space.
INTERIOR_SPACE_VALUE_IN_FUNCTION = 0

# INTERIOR_SPACE_VALUE_IN_CONNECTION_INPUT: Value used to mark interior spaces
# (the areas to be connected) when preparing an array for the
# `cv2.connectedComponentsWithStats` function. Typically, this is 1 (foreground).
INTERIOR_SPACE_VALUE_IN_CONNECTION_INPUT = 1

# EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT: Number of Control Volume units by which
# the exterior wall representation is "thickened" or expanded inwards during
# floor plan processing. This helps define a more robust exterior wall layer.
EXPAND_EXTERIOR_WALLS_BY_CV_AMOUNT = 2

# GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT: Value used to mark areas that are
# *not* part of the components to be connected when preparing an array for
# `cv2.connectedComponentsWithStats`. This typically represents walls and
# exterior space, often set to 0 (background).
GENERIC_SPACE_VALUE_IN_CONNECTION_INPUT = 0

# WALLS_AND_EXPANDED_BOOLS: A threshold value used in `enlarge_exterior_walls`
# to identify CVs that are part of either original walls or the expanded
# exterior wall region during processing.
WALLS_AND_EXPANDED_BOOLS = 2 # Assumes binary masks are summed; 2 means it was wall + part of expansion or overlapping walls.

# INTERIOR_WALL_VALUE_IN_FILE_INPUT: The integer value expected in raw input
# floor plan files to designate interior walls.
INTERIOR_WALL_VALUE_IN_FILE_INPUT = 1


# --- String Labels and Names ---

# EXTERIOR_SPACE_NAME_IN_ROOM_DICT: String key used in the `room_dict`
# (mapping room/zone names to CV coordinates) to identify the collection of
# CVs belonging to the exterior space.
EXTERIOR_SPACE_NAME_IN_ROOM_DICT = "exterior_space"

# INTERIOR_WALL_NAME_IN_ROOM_DICT: String key used in the `room_dict` to
# identify CVs that are part of interior walls, if they are explicitly stored.
INTERIOR_WALL_NAME_IN_ROOM_DICT = "interior_wall"

# LABEL_FOR_WALLS: String label used in the `cv_type` array to mark
# Control Volumes that are classified as walls (either interior or exterior).
LABEL_FOR_WALLS = "wall"

# LABEL_FOR_INTERIOR_SPACE: String label used in the `cv_type` array for CVs
# classified as interior air/space within rooms or zones.
LABEL_FOR_INTERIOR_SPACE = "interior_space"

# LABEL_FOR_EXTERIOR_SPACE: String label used in the `cv_type` array for CVs
# classified as exterior space or ambient air.
LABEL_FOR_EXTERIOR_SPACE = "exterior_space"

# ROOM_STRING_DESIGNATOR: Prefix used for naming rooms/zones when they are
# automatically generated from connected components (e.g., "room_1", "room_2").
ROOM_STRING_DESIGNATOR = "room"


# --- Simulation Parameters and Paths ---

# VIDEO_PATH_ROOT: Default root directory path for saving video logs or
# visualizations generated by the building simulation.
# Note: This is a CNS path specific to Google's infrastructure.
VIDEO_PATH_ROOT = "/cns/oz-d/home/smart-buildings-control-team/smart-buildings/geometric_sim_videos/"

# WATT_LIMIT: A threshold in Watts, potentially used to limit the thermal power
# output of diffusers or other HVAC equipment in the simulation.
WATT_LIMIT = 500.0 # Changed to float for consistency if used with float power values
