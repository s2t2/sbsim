"""Utilities for plotting building simulation data and creating videos.

This module provides functions for visualizing various aspects of a smart
building simulation, including:
- Temperature distributions within the building (planform view).
- Time-series plots of zone temperatures, ambient temperature, and setpoints.
- Time-series plots of HVAC energy consumption rates.

It also includes helper functions for managing metrics collected during a
simulation run and for combining individual plot frames into a video.
"""

import collections
import copy # For deepcopy in BuildingRenderer.render
import functools
import io
import os
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Tuple # Added Mapping, Tuple

from matplotlib import cm
from matplotlib import patches
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import PIL # For Image, ImageDraw, ImageFont
from PIL import ImageDraw
import seaborn as sn # type: ignore[import-untyped]

# Assuming building_utils and constants are in smart_control.simulator
from smart_control.simulator import building_utils
from smart_control.simulator import constants as sim_constants
# Assuming setpoint_schedule is in smart_control.simulator
from smart_control.simulator import setpoint_schedule as schedule_py


# TODO(b/260300338): Replace K_TO_C with constants.KELVIN_TO_CELSIUS_OFFSET
# from smart_control.utils.constants import KELVIN_TO_CELSIUS_OFFSET
K_TO_C: float = 273.15


def get_temp_colors(min_k: int, max_k: int) -> Dict[int, Any]:
  """Generates a discrete color map for temperatures in Kelvin.

  Creates a dictionary mapping integer Kelvin temperatures within the given
  range to RGBA color values from a 'rainbow' colormap.

  Args:
    min_k (int): The minimum temperature in Kelvin for the color map.
    max_k (int): The maximum temperature in Kelvin for the color map.

  Returns:
    Dict[int, Any]: A dictionary mapping integer Kelvin temperatures to
    Matplotlib RGBA color tuples.

  Raises:
    ValueError: If `min_k` is greater than `max_k`.

  Example:
    >>> colors = get_temp_colors(290, 292)
    >>> print(colors[290]) # Output will be an RGBA tuple like (r, g, b, a)
  """

  def get_temp_color_palette(num_colors: int) -> np.ndarray:
    """Helper to get a segment of the rainbow colormap."""
    # The original calculation for ys seems arbitrary and doesn't directly
    # influence the color selection from linspace.
    # Using a simple linspace for color selection.
    return cm.get_cmap("rainbow")(np.linspace(0, 1, num_colors))

  if min_k > max_k:
    raise ValueError("min_k cannot be greater than max_k.")
  num_colors = int(max_k - min_k + 1)
  colors_array = get_temp_color_palette(num_colors)
  temp_color_map: Dict[int, Any] = {}
  for i, temp_val_k in enumerate(range(min_k, max_k + 1)):
    temp_color_map[temp_val_k] = colors_array[i]
  return temp_color_map


def render_building_subplot(
    ax: plt.Axes,
    min_render_temp_k: int,
    max_render_temp_k: int,
    ambient_temp_k: float,
    building_model: Any, # Ideally: building.Building or building.FloorPlanBasedBuilding
    current_time: pd.Timestamp,
) -> None:
  """Renders a planform (top-down) view of the building's thermal state.

  This function draws the building layout, color-coding Control Volumes (CVs)
  by temperature, and optionally overlays diffuser states and zone information
  onto a Matplotlib Axes object.

  Note: This function assumes `building_model` has specific attributes like
  `cv_size_cm`, `room_shape`, `building_shape` (for grid-based), `temp`,
  `conductivity`, `input_q`, `diffusers`, and method `get_zone_temp_stats`.
  This makes it tightly coupled to a particular `Building` class structure.
  Adaptation might be needed for `FloorPlanBasedBuilding` if its attributes differ.

  Args:
    ax (plt.Axes): The Matplotlib Axes object on which to render the building.
    min_render_temp_k (int): Minimum temperature (K) for color scaling.
    max_render_temp_k (int): Maximum temperature (K) for color scaling.
    ambient_temp_k (float): Current ambient outdoor temperature (K).
    building_model (Any): An instance of the building model (e.g.,
      `smart_control.simulator.building.Building`).
    current_time (pd.Timestamp): The current simulation timestamp.
  """
  # Building and zone dimension calculations
  # These are highly specific to the grid-based `Building` class structure.
  # For a generic renderer, these would need to be obtained differently or
  # the function adapted for various building model types.
  cv_size_m = building_model.cv_size_cm / 100.0

  # Determine total CVs based on building_model type or available attributes
  if hasattr(building_model, 'room_shape') and hasattr(building_model, 'building_shape'): # Grid-based
    total_cv_rows = (building_model.room_shape[0] + 1) * \
                    building_model.building_shape[0] + 3
    total_cv_cols = (building_model.room_shape[1] + 1) * \
                    building_model.building_shape[1] + 3
  elif hasattr(building_model, 'floor_plan'): # FloorPlanBasedBuilding
    total_cv_rows, total_cv_cols = building_model.floor_plan.shape
  else:
    raise ValueError("Unsupported building_model type or missing shape attributes.")

  # Denominators for normalizing coordinates to [0,1] for transAxes
  # Using total CV dimensions * CV size for normalization
  denom_x = total_cv_cols * cv_size_m
  denom_y = total_cv_rows * cv_size_m
  if denom_x == 0 or denom_y == 0: # Avoid division by zero
      return


  temp_color_map = get_temp_colors(min_render_temp_k, max_render_temp_k)

  def get_bounded_temp_color(temp_k: float) -> Any:
    """Clips temperature and gets corresponding color."""
    bounded_temp = int(np.clip(temp_k, min_render_temp_k, max_render_temp_k))
    return temp_color_map.get(bounded_temp, temp_color_map[min_render_temp_k])

  # Render ambient background (full axes)
  ambient_color = get_bounded_temp_color(ambient_temp_k)
  ambient_rect = plt.Rectangle(
      (0, 0), 1, 1, fill=True, edgecolor=None, alpha=0.6,
      facecolor=ambient_color, transform=ax.transAxes, clip_on=False
  )
  ax.add_patch(ambient_rect)

  # White rectangle for the building footprint area before CVs
  # This logic is specific and creates a border effect.
  # For FloorPlanBasedBuilding, this might need to use actual floor plan boundaries.
  # Assuming the inner area (excluding 1 CV border for ambient rendering)
  building_area_left_norm = cv_size_m / denom_x
  building_area_bottom_norm = cv_size_m / denom_y
  building_area_width_norm = (total_cv_cols - 2) * cv_size_m / denom_x
  building_area_height_norm = (total_cv_rows - 2) * cv_size_m / denom_y

  building_bg_rect = plt.Rectangle(
      (building_area_left_norm, building_area_bottom_norm),
      building_area_width_norm, building_area_height_norm,
      fill=True, edgecolor=None, alpha=1.0, facecolor="white",
      transform=ax.transAxes, clip_on=False
  )
  ax.add_patch(building_bg_rect)


  # Render individual Control Volumes
  for r_idx in range(total_cv_rows):
    for c_idx in range(total_cv_cols):
      # CV's bottom-left corner in normalized coordinates (plotting (c,r))
      cv_left_norm = (c_idx * cv_size_m) / denom_x
      cv_bottom_norm = (r_idx * cv_size_m) / denom_y # Inverted for typical image plot
      cv_width_norm = cv_size_m / denom_x
      cv_height_norm = cv_size_m / denom_y

      # This part of original rendering logic for edges was complex and
      # specific to the old CV indexing (0,0 at bottom-left).
      # Modern Matplotlib usually handles (row,col) from top-left.
      # The original code's `render_control_volume` had complex adjustments for edges.
      # For simplicity, we'll draw each CV as a square.
      # The distinction between edge/corner CVs is mainly for FDM, not render rect size.

      temp_k = building_model.temp[r_idx, c_idx]
      cv_color = get_bounded_temp_color(temp_k)
      # Determine edge color based on conductivity (wall type)
      cond = building_model.conductivity[r_idx, c_idx]
      edge_color = "lightgray" # Default for air/interior
      if cond < 0.1: edge_color = "black"     # Strong wall (e.g., exterior)
      elif cond < 5.0: edge_color = "dimgray" # Weaker wall (e.g., interior)

      cv_rect = plt.Rectangle(
          (cv_left_norm, cv_bottom_norm), cv_width_norm, cv_height_norm,
          fill=True, edgecolor=edge_color, alpha=0.6, facecolor=cv_color,
          transform=ax.transAxes, clip_on=False
      )
      ax.add_patch(cv_rect)

  # Render zone temperature statistics (specific to grid-based Building)
  if hasattr(building_model, 'room_shape') and hasattr(building_model, 'building_shape'):
    zone_cv_rows = building_model.room_shape[0] + 1
    zone_cv_cols = building_model.room_shape[1] + 1
    for zr_idx in range(building_model.building_shape[0]): # zone row
      for zc_idx in range(building_model.building_shape[1]): # zone col
        # Position for text, normalized. Needs careful adjustment.
        text_l_norm = (zc_idx * zone_cv_cols * cv_size_m + cv_size_m) / denom_x
        text_b_norm = (zr_idx * zone_cv_rows * cv_size_m + cv_size_m) / denom_y
        text_h_norm = zone_cv_rows * cv_size_m / denom_y
        t_min, t_max, t_avg = building_model.get_zone_temp_stats((zr_idx, zc_idx))
        label = (f"Zone ({zr_idx},{zc_idx})\n"
                   f"Min: {t_min - K_TO_C:.1f}°C\n"
                   f"Max: {t_max - K_TO_C:.1f}°C\n"
                   f"Avg: {t_avg - K_TO_C:.1f}°C")
        ax.text(0.01 + text_l_norm, text_b_norm + text_h_norm - 0.02, label,
                  ha="left", va="top", transform=ax.transAxes, fontsize=8,
                  color="black", bbox=dict(facecolor='white', alpha=0.5, pad=1))

  # Render diffusers
  if hasattr(building_model, "input_q") and hasattr(building_model, "diffusers"):
    for r_idx in range(total_cv_rows):
      for c_idx in range(total_cv_cols):
        if building_model.diffusers[r_idx, c_idx] > 0:
          # Diffuser location in normalized coordinates
          diff_x_norm = (c_idx + 0.5) * cv_size_m / denom_x
          diff_y_norm = (r_idx + 0.5) * cv_size_m / denom_y
          # Diffuser size (relative to CV size)
          diff_h_norm = 0.5 * cv_size_m / denom_y
          diff_w_norm = 0.5 * cv_size_m / denom_x

          heat_val = building_model.input_q[r_idx, c_idx]
          diff_color = "gray"
          if heat_val > 1e-3: diff_color = "red" # Heating
          elif heat_val < -1e-3: diff_color = "blue" # Cooling

          ellipse = patches.Ellipse((diff_x_norm, diff_y_norm), width=diff_w_norm, height=diff_h_norm,
                                   facecolor=diff_color, alpha=1.0, edgecolor="gray",
                                   transform=ax.transAxes, clip_on=False)
          ax.add_patch(ellipse)
          if abs(heat_val) > 1e-3: # Only label if significant
            ax.text(diff_x_norm + 0.005, diff_y_norm, f"{heat_val / 1000.0:.1f}kW",
                      ha="left", va="center", transform=ax.transAxes,
                      fontsize=7, color="white",
                      bbox=dict(facecolor='black', alpha=0.5, pad=0.5))

  # Add overall plot labels (timestamp, ambient temp)
  time_label = (f"Time: {current_time.strftime('%Y-%m-%d %H:%M')}\n"
                f"Ambient: {ambient_temp_k - K_TO_C:.1f}°C")
  ax.text(0.01, 0.99, time_label, ha="left", va="top",
            transform=ax.transAxes, fontsize=12, color="black",
            bbox=dict(facecolor='white', alpha=0.7, pad=2))
  ax.axis("off") # Turn off a-is numbers and ticks for the building plot


def plot_zone_temp_timeline(
    ax: plt.Axes,
    schedule: Any, # Should be setpoint_schedule.SetpointSchedule
    temps_timeseries_df: pd.DataFrame,
    end_timestamp: pd.Timestamp
) -> None:
  """Plots zone temperatures, setpoints, and ambient temperature over time.

  Args:
    ax (plt.Axes): Matplotlib Axes to plot on.
    schedule (Any): A `SetpointSchedule` object to get temperature windows.
    temps_timeseries_df (pd.DataFrame): DataFrame where index is Timestamp,
      and columns include zone temperatures (e.g., 'Zone1_Temp') and
      'ambient' temperature. Temperatures are assumed to be in Kelvin.
    end_timestamp (pd.Timestamp): The end timestamp for the plot's x-axis.
  """
  # Plot setpoint windows (deadbands)
  setpoint_windows = schedule.get_plot_data(
      temps_timeseries_df.index.min(), end_timestamp
  )
  for _, row_data in setpoint_windows.iterrows():
    start_num = mdates.date2num(row_data["start_time"])
    end_num = mdates.date2num(row_data["end_time"])
    heating_sp_c = row_data["heating_setpoint"] - K_TO_C
    cooling_sp_c = row_data["cooling_setpoint"] - K_TO_C
    deadband_rect = plt.Rectangle(
        (start_num, heating_sp_c), end_num - start_num,
        cooling_sp_c - heating_sp_c,
        fill=True, edgecolor=None, alpha=0.3, facecolor="lightgrey"
    )
    ax.add_patch(deadband_rect)

  # Plot zone temperatures
  zone_temp_cols = [col for col in temps_timeseries_df.columns if col != "ambient"]
  for zone_col in zone_temp_cols:
    ax.plot(
        temps_timeseries_df.index, temps_timeseries_df[zone_col] - K_TO_C,
        color="yellow", marker=None, alpha=0.8, lw=1, linestyle="-",
        label=f"{zone_col} Temp" if len(zone_temp_cols) < 5 else None # Avoid crowded legend
    )
  if len(zone_temp_cols) >= 5: # Add one label for all zones if many
      ax.plot([],[], color='yellow', lw=1, label='Zone Temps')


  # Plot ambient temperature
  ax.plot(
      temps_timeseries_df.index, temps_timeseries_df["ambient"] - K_TO_C,
      color="blue", marker=None, alpha=1, lw=2, linestyle="-", label="Ambient"
  )

  # Formatting
  ax.set_facecolor("black")
  ax.xaxis.tick_top() # X-axis labels at the top
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M")) # Compact format
  ax.grid(color="gray", linestyle="-", linewidth=0.5)
  ax.set_ylabel("Temperature (°C)", color="blue", fontsize=12)
  ax.set_xlim(left=temps_timeseries_df.index.min(), right=end_timestamp)
  ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
  ax.legend(fontsize=10)


def plot_energy_rates_timeline(
    ax: plt.Axes, energy_rates_df: pd.DataFrame, end_timestamp: pd.Timestamp
) -> None:
  """Plots HVAC component energy consumption rates over time.

  Args:
    ax (plt.Axes): Matplotlib Axes to plot on.
    energy_rates_df (pd.DataFrame): DataFrame with energy rates (Watts) for
      components like 'boiler_thermal_energy_rate',
      'air_handler_thermal_energy_rate', etc. Index is Timestamp.
    end_timestamp (pd.Timestamp): The end timestamp for the plot's x-axis.
  """
  # Plot boiler energy rates (thermal in solid, electrical in dashed)
  if "boiler_thermal_energy_rate" in energy_rates_df:
    ax.plot(
        energy_rates_df.index,
        energy_rates_df["boiler_thermal_energy_rate"] / 1000.0, # W to kW
        color="lime", lw=2, linestyle="-", label="Boiler Thermal (Gas)"
    )
  if "boiler_electrical_energy_rate" in energy_rates_df:
    ax.plot(
        energy_rates_df.index,
        energy_rates_df["boiler_electrical_energy_rate"] / 1000.0,
        color="green", lw=2, linestyle="--", label="Boiler Pump Elec."
    )

  # Plot air handler energy rates
  if "air_handler_thermal_energy_rate" in energy_rates_df: # Cooling/Heating coil
    ax.plot(
        energy_rates_df.index,
        energy_rates_df["air_handler_thermal_energy_rate"] / 1000.0,
        color="magenta", lw=2, linestyle="-", label="AHU Coil (Elec.)"
    )
  fan_power_kw = pd.Series(0.0, index=energy_rates_df.index)
  if "air_handler_intake_fan_energy_rate" in energy_rates_df:
      fan_power_kw += energy_rates_df["air_handler_intake_fan_energy_rate"]
  if "air_handler_exhaust_fan_energy_rate" in energy_rates_df:
      fan_power_kw += energy_rates_df["air_handler_exhaust_fan_energy_rate"]
  if not fan_power_kw.empty:
      ax.plot(
          energy_rates_df.index, fan_power_kw / 1000.0,
          color="purple", lw=2, linestyle="--", label="AHU Fans (Elec.)"
      )

  # Formatting
  ax.set_facecolor("black")
  ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
  ax.grid(color="gray", linestyle="-", linewidth=0.5)
  ax.set_ylabel("Energy Rate (kW)", color="blue", fontsize=12)
  ax.set_xlim(left=energy_rates_df.index.min(), right=end_timestamp)
  ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
  ax.legend(fontsize=10)


def plot_combined_results(
    temps_timeseries_df: pd.DataFrame,
    energy_rates_df: pd.DataFrame,
    min_k: int,
    max_k: int,
    ambient_temp_k: float,
    building_model: Any, # building.Building or similar
    schedule: Any, # setpoint_schedule.SetpointSchedule
    current_sim_time: pd.Timestamp,
    episode_end_time: pd.Timestamp,
    output_dir_path: Optional[str] = None,
) -> None:
  """Creates and optionally saves a combined plot of simulation results.

  The plot includes:
  1. A planform view of the building's current thermal state.
  2. A time-series plot of zone temperatures, setpoints, and ambient temp.
  3. A time-series plot of HVAC energy consumption rates.

  Args:
    temps_timeseries_df (pd.DataFrame): DataFrame of temperature time series.
    energy_rates_df (pd.DataFrame): DataFrame of energy rate time series.
    min_k (int): Min temperature (K) for building render color scale.
    max_k (int): Max temperature (K) for building render color scale.
    ambient_temp_k (float): Current ambient temperature (K).
    building_model (Any): The building model instance.
    schedule (Any): The setpoint schedule instance.
    current_sim_time (pd.Timestamp): Current simulation timestamp.
    episode_end_time (pd.Timestamp): End timestamp for time series plots.
    output_dir_path (Optional[str]): If provided, directory to save the plot image.
      If None, the plot is displayed using `plt.show()`.
  """
  fig, (ax_temps, ax_energy, ax_building) = plt.subplots(
      nrows=3, ncols=1,
      gridspec_kw={"height_ratios": [1, 1, 2.3]}, # Relative heights
      figsize=(20, 25) # Larger figure size
  )
  plt.subplots_adjust(hspace=0.3) # Add space between subplots

  plot_zone_temp_timeline(ax_temps, schedule, temps_timeseries_df, episode_end_time)
  ax_temps.set_title("Zone and Ambient Temperatures vs. Setpoints", fontsize=16)

  plot_energy_rates_timeline(ax_energy, energy_rates_df, episode_end_time)
  ax_energy.set_title("HVAC Energy Consumption Rates", fontsize=16)

  render_building_subplot(
      ax_building, min_k, max_k, ambient_temp_k, building_model, current_sim_time
  )
  ax_building.set_title(f"Building Thermal Map at {current_sim_time.strftime('%Y-%m-%d %H:%M')}", fontsize=16)


  if output_dir_path:
    filename = f"sim_step_{current_sim_time.strftime('%Y%m%d_%H%M%S')}.png"
    full_path = os.path.join(output_dir_path, filename)
    try:
      pathlib.Path(output_dir_path).mkdir(parents=True, exist_ok=True)
      plt.savefig(full_path, bbox_inches="tight")
      logging.info("Saved combined plot to %s", full_path)
    except IOError as e:
      logging.error("Failed to save plot to %s: %s", full_path, e)
    plt.close(fig) # Close figure to free memory when saving
  else:
    plt.show()


def init_metrics() -> Dict[str, List[Any]]:
  """Initializes a dictionary to store various simulation metrics.

  Returns:
    Dict[str, List[Any]]: An empty dictionary structured to hold lists of
    metric values collected during a simulation run. Keys include:
    'timestamps', 'ambient_temps', 'avg_temps_timeseries' (itself a
    defaultdict(list)), 'boiler_thermal_energy_rates', etc.
  """
  metrics: Dict[str, List[Any]] = {
      "timestamps": [],
      "ambient_temps": [],
      "avg_temps_timeseries": collections.defaultdict(list), # Zone temps
      "boiler_thermal_energy_rates": [],
      "boiler_electrical_energy_rates": [],
      "air_handler_intake_fan_energy_rates": [],
      "air_handler_exhaust_fan_energy_rates": [],
      "air_handler_thermal_energy_rates": [], # For AHU heating/cooling coil
  }
  return metrics


def update_metrics(
    metrics_dict: Dict[str, List[Any]],
    current_sim_timestamp: pd.Timestamp,
    current_ambient_temp_k: float,
    # `building_model` and `hvac_model` should be more specific types if possible
    building_model: Any, # e.g., building.Building
    hvac_model: Any,     # e.g., hvac.Hvac
) -> Dict[str, List[Any]]:
  """Updates the metrics dictionary with data from the current simulation step.

  Args:
    metrics_dict (Dict[str, List[Any]]): The dictionary holding metric lists.
    current_sim_timestamp (pd.Timestamp): Current simulation time.
    current_ambient_temp_k (float): Current ambient temperature (K).
    building_model (Any): The building model instance to get zone temperatures.
    hvac_model (Any): The HVAC model instance to get energy rates.

  Returns:
    Dict[str, List[Any]]: The updated metrics dictionary.
  """
  metrics_dict["timestamps"].append(current_sim_timestamp)
  metrics_dict["ambient_temps"].append(current_ambient_temp_k)

  # Update average zone temperatures
  # Assuming get_zone_average_temps returns {zone_id: avg_temp}
  for zone_id, avg_temp_k in building_model.get_zone_average_temps().items():
    metrics_dict["avg_temps_timeseries"][zone_id].append(avg_temp_k)

  # HVAC energy metrics
  # Boiler
  # Note: The original `supply_air_temp` argument to boiler.compute_thermal_energy_rate
  # was likely incorrect. It should be return_water_temp and ambient_temp.
  # Assuming boiler has `return_water_temperature_sensor` attribute.
  # This part needs correct attributes from the actual Boiler class.
  # Placeholder logic if attributes differ:
  boiler_return_temp_k = getattr(hvac_model.boiler, "return_water_temperature_sensor", current_ambient_temp_k) # Default if not found
  metrics_dict["boiler_thermal_energy_rates"].append(
      hvac_model.boiler.compute_thermal_energy_rate(
          return_water_temp_k=boiler_return_temp_k, # Placeholder
          ambient_temp_k=current_ambient_temp_k # For losses
      )
  )
  metrics_dict["boiler_electrical_energy_rates"].append(
      hvac_model.boiler.compute_pump_power() # Assuming this is in Watts
  )

  # Air Handler
  # compute_thermal_energy_rate needs recirc_temp and ambient_temp.
  # Assuming building_model.temp.mean() is a proxy for recirc_temp.
  recirc_temp_k = building_model.temp.mean() if hasattr(building_model, 'temp') else current_ambient_temp_k
  metrics_dict["air_handler_intake_fan_energy_rates"].append(
      hvac_model.air_handler.compute_intake_fan_energy_rate()
  )
  metrics_dict["air_handler_exhaust_fan_energy_rates"].append(
      hvac_model.air_handler.compute_exhaust_fan_energy_rate()
  )
  metrics_dict["air_handler_thermal_energy_rates"].append(
      hvac_model.air_handler.compute_thermal_energy_rate(
          recirculation_temp_k=recirc_temp_k, # Placeholder for actual recirc
          ambient_temp_k=current_ambient_temp_k
      )
  )
  return metrics_dict


def plot_update(
    metrics_dict: Dict[str, List[Any]],
    current_ambient_temp_k: float,
    building_model: Any, # building.Building
    schedule_obj: Any,   # setpoint_schedule.SetpointSchedule
    current_sim_timestamp: pd.Timestamp,
    episode_end_timestamp: pd.Timestamp,
    image_output_directory: Optional[str] = None,
) -> None:
  """Generates and displays or saves a combined plot of current simulation metrics.

  Args:
    metrics_dict (Dict[str, List[Any]]): Dictionary of collected metrics.
    current_ambient_temp_k (float): Current ambient temperature (K).
    building_model (Any): The building model instance.
    schedule_obj (Any): The setpoint schedule instance.
    current_sim_timestamp (pd.Timestamp): Current simulation time.
    episode_end_timestamp (pd.Timestamp): End time for the x-axis of plots.
    image_output_directory (Optional[str]): Directory to save the plot image.
      If None, displays the plot interactively.
  """
  if not metrics_dict["timestamps"]:
    logging.warning("No data in metrics_dict to plot.")
    return

  # Prepare DataFrame for temperatures
  temps_df = pd.DataFrame(index=pd.to_datetime(metrics_dict["timestamps"]))
  temps_df["ambient"] = metrics_dict["ambient_temps"]
  # Add zone temperatures
  for zone_id, zone_temps_list in metrics_dict["avg_temps_timeseries"].items():
    # Ensure list length matches timestamps if data collection was interrupted
    temps_df[str(zone_id)] = pd.Series(zone_temps_list, index=temps_df.index[:len(zone_temps_list)])


  # Prepare DataFrame for energy rates
  energy_df = pd.DataFrame(index=pd.to_datetime(metrics_dict["timestamps"]))
  energy_df["boiler_thermal_energy_rate"] = metrics_dict["boiler_thermal_energy_rates"]
  energy_df["boiler_electrical_energy_rate"] = metrics_dict["boiler_electrical_energy_rates"]
  energy_df["air_handler_intake_fan_energy_rate"] = metrics_dict["air_handler_intake_fan_energy_rates"]
  energy_df["air_handler_exhaust_fan_energy_rate"] = metrics_dict["air_handler_exhaust_fan_energy_rates"]
  energy_df["air_handler_thermal_energy_rate"] = metrics_dict["air_handler_thermal_energy_rates"]

  plot_combined_results(
      temps_timeseries_df=temps_df,
      energy_rates_df=energy_df,
      min_k=280, # Example value, could be configurable
      max_k=305, # Example value
      ambient_temp_k=current_ambient_temp_k,
      building_model=building_model,
      schedule=schedule_obj,
      current_sim_time=current_sim_timestamp,
      episode_end_time=episode_end_timestamp,
      output_dir_path=image_output_directory,
  )
