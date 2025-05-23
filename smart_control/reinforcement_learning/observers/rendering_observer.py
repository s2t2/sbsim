"""Observer for rendering environment state and plotting metrics.

This module provides the `RenderingObserver`, an implementation of the `Observer`
interface. It is designed to:
1. Periodically render the state of the simulation environment (if a render
   function is provided or the environment supports it).
2. Generate and save plots of various time-series metrics collected during an
   episode, such as rewards, energy consumption, temperatures, and actions.
"""

import logging
import os
from typing import Callable, Optional, Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import pytz # type: ignore[import-untyped]
from tf_agents.environments import py_environment
from tf_agents.trajectories import trajectory as trajectory_lib

# Assuming environment.Environment is the specific type from this project.
from smart_control.environment import environment as smart_control_environment
from smart_control.reinforcement_learning.observers.base_observer import Observer
from smart_control.reinforcement_learning.utils.config import RENDERS_PATH
from smart_control.reinforcement_learning.utils.constants import DEFAULT_TIME_ZONE
from smart_control.reinforcement_learning.utils.constants import KELVIN_TO_CELSIUS as _KELVIN_TO_CELSIUS
from smart_control.reinforcement_learning.utils.data_processing import get_action_timeseries
from smart_control.reinforcement_learning.utils.data_processing import get_energy_timeseries
from smart_control.reinforcement_learning.utils.data_processing import get_latest_episode_reader
from smart_control.reinforcement_learning.utils.data_processing import get_outside_air_temperature_timeseries
from smart_control.reinforcement_learning.utils.data_processing import get_reward_timeseries
from smart_control.reinforcement_learning.utils.data_processing import get_zone_timeseries
from smart_control.utils import building_renderer
from smart_control.utils import data_reader

logger = logging.getLogger(__name__)


class RenderingObserver(Observer):
  """Renders environment state and plots metrics at specified intervals.

  This observer is called at each step of the agent-environment interaction.
  It periodically triggers:
  - Rendering of the environment's current state (e.g., a visual
    representation of the building).
  - Plotting of time-series data collected during the episode, such as
    rewards, energy usage, temperatures, and agent actions.

  Rendered images and plots are saved to a specified directory.

  Attributes:
    KELVIN_TO_CELSIUS (float): Constant for converting Kelvin to Celsius.
    _counter (int): Number of trajectories processed since the last reset.
    _render_interval_steps (int): Frequency (in steps) for rendering/plotting.
    _environment (Optional[py_environment.PyEnvironment]): The environment.
    _render_fn (Optional[Callable]): Custom function for environment rendering.
    _plot_fn (Optional[Callable]): Custom function for plotting (not used in
      current impl, plotting is hardcoded).
    _clear_output_before_render (bool): (Not actively used) Intended for
      clearing output in interactive environments.
    _time_zone (str): Time zone for displaying timestamps in plots.
    _cumulative_reward (float): Reward accumulated since the last reset.
    _start_time (Optional[pd.Timestamp]): Wall clock time of the first call
      after a reset.
    _save_path (str): Directory to save rendered images and plots.
    _num_timesteps_in_episode (int): Total timesteps in an episode, if known.
  """

  KELVIN_TO_CELSIUS: float = _KELVIN_TO_CELSIUS

  def __init__(
      self,
      render_interval_steps: int = 100,
      env: Optional[py_environment.PyEnvironment] = None,
      render_fn: Optional[Callable[..., Any]] = None,
      plot_fn: Optional[Callable[..., Any]] = None,
      clear_output_before_render: bool = True, # pylint: disable=unused-argument
      time_zone: str = DEFAULT_TIME_ZONE,
      save_path: str = RENDERS_PATH,
  ):
    """Initializes the RenderingObserver.

    Args:
      render_interval_steps (int): Number of agent steps between each rendering
        and plotting action.
      env (Optional[py_environment.PyEnvironment]): The TF-Agents Python
        environment. This is used to access simulation data for plotting and
        rendering.
      render_fn (Optional[Callable[..., Any]]): A custom function to render the
        environment. If None, the observer will try to use a default rendering
        mechanism based on `smart_control.utils.building_renderer`.
      plot_fn (Optional[Callable[..., Any]]): A custom function for plotting
        metrics. If None, the observer uses its internal plotting methods.
        (Currently, internal methods are always used).
      clear_output_before_render (bool): If True, attempts to clear previous
        output before rendering. (Note: This has limited effect in non-
        interactive environments like typical script executions).
      time_zone (str): The time zone to use for formatting timestamps in plots.
      save_path (str): The directory path where rendered images and plots will
        be saved.
    """
    self._counter: int = 0
    self._render_interval_steps: int = render_interval_steps
    self._environment: Optional[py_environment.PyEnvironment] = env
    self._render_fn: Optional[Callable[..., Any]] = render_fn
    self._plot_fn: Optional[Callable[..., Any]] = plot_fn
    # self._clear_output_before_render = clear_output_before_render # Not used
    self._time_zone: str = time_zone
    self._cumulative_reward: float = 0.0
    self._start_time: Optional[pd.Timestamp] = None
    self._save_path: str = save_path
    self._num_timesteps_in_episode: int = 0

    os.makedirs(self._save_path, exist_ok=True)

    if self._environment and hasattr(self._environment, "pyenv"):
      # Accessing underlying PyEnvironment, common in TF-Agents wrappers.
      # Assumes the first env in a batched env has relevant episode info.
      try:
        self._num_timesteps_in_episode = self._environment.pyenv.envs[
            0
        ]._num_timesteps_in_episode
      except (AttributeError, IndexError) as e:
        logger.warning(
            "Could not determine _num_timesteps_in_episode from env: %s", e
        )
        self._num_timesteps_in_episode = -1 # Indicates unknown
    else:
        self._num_timesteps_in_episode = -1


  def _format_plot(
      self,
      ax: plt.Axes,
      ylabel: str,
      start_time_utc: pd.Timestamp,
      end_time_utc: pd.Timestamp,
      time_zone_str: str,
  ) -> None:
    """Applies common formatting to a Matplotlib Axes object for time series.

    Args:
      ax (plt.Axes): The Matplotlib Axes object to format.
      ylabel (str): The label for the Y-axis.
      start_time_utc (pd.Timestamp): The UTC start time for the X-axis limit.
      end_time_utc (pd.Timestamp): The UTC end time for the X-axis limit.
      time_zone_str (str): The string representation of the target time zone
        (e.g., 'America/Los_Angeles').
    """
    ax.set_facecolor("black")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%a %m/%d %H:%M", tz=pytz.timezone(time_zone_str))
    )
    ax.grid(color="gray", linestyle="-", linewidth=1.0)
    ax.set_ylabel(ylabel, color="blue", fontsize=12)
    # Ensure timestamps are timezone-aware for correct limiting
    ax.set_xlim(left=start_time_utc, right=end_time_utc)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(prop={"size": 10})

  def _plot_reward_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone_str: str
  ) -> None:
    """Plots the cumulative reward over time.

    Args:
      ax (plt.Axes): The Matplotlib Axes to plot on.
      reward_timeseries (pd.DataFrame): DataFrame with a 'cumulative_reward'
        column and a DatetimeIndex.
      time_zone_str (str): Target time zone for display.
    """
    # Convert UTC index to local time for plotting if index is tz-aware UTC
    if reward_timeseries.index.tz is not None:
        local_times = reward_timeseries.index.tz_convert(time_zone_str)
    else: # If tz-naive, assume UTC and localize
        local_times = reward_timeseries.index.tz_localize('UTC').tz_convert(time_zone_str)


    ax.plot(
        local_times,
        reward_timeseries["cumulative_reward"],
        color="royalblue",
        marker=None,
        alpha=1,
        lw=6,
        linestyle="-",
        label="reward",
    )
    self._format_plot(
        ax,
        "Agent Reward",
        reward_timeseries.index.min(),
        reward_timeseries.index.max(),
        time_zone_str,
    )

  def _plot_energy_timeline(
      self,
      ax: plt.Axes,
      energy_timeseries: pd.DataFrame,
      time_zone_str: str,
      cumulative: bool = False,
  ) -> None:
    """Plots energy consumption timelines for different HVAC components.

    Args:
      ax (plt.Axes): The Matplotlib Axes to plot on.
      energy_timeseries (pd.DataFrame): DataFrame containing energy data,
        expected to have columns like 'device_type', 'start_time',
        'end_time', and various energy rate columns.
      time_zone_str (str): Target time zone for display.
      cumulative (bool): If True, plot cumulative energy (kWh), otherwise
        plot instantaneous power (kW).
    """

    def _to_kwh(
        energy_rate_watts: pd.Series,
        step_interval: pd.Timedelta = pd.Timedelta(5, unit="minutes"),
    ) -> pd.Series:
      """Converts power (Watts) over an interval to energy (kWh)."""
      kw_power = energy_rate_watts / 1000.0
      # Energy = Power * Time. Time is step_interval in hours.
      energy_kwh_per_step = kw_power * (
          step_interval.total_seconds() / 3600.0
      )
      return energy_kwh_per_step.cumsum()

    # Plot Air Handler energy
    ah_energy = energy_timeseries[
        energy_timeseries["device_type"] == "air_handler"
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    if ah_energy.empty:
        logger.warning("No air handler energy data to plot.")
    else:
        # Ensure start_time is datetime before plotting
        ah_energy["start_time"] = pd.to_datetime(ah_energy["start_time"])
        if ah_energy["start_time"].dt.tz is None: # If naive, assume UTC
            ah_energy["start_time"] = ah_energy["start_time"].dt.tz_localize('UTC')
        ah_plot_times = ah_energy["start_time"].dt.tz_convert(time_zone_str)

        ac_rate_col = "air_handler_air_conditioner_energy_rate"
        blower_rate_col = "air_handler_blower_electrical_energy_rate"

        if cumulative:
            ac_plot_data = _to_kwh(ah_energy[ac_rate_col])
            blower_plot_data = _to_kwh(ah_energy[blower_rate_col])
        else:
            ac_plot_data = ah_energy[ac_rate_col] / 1000.0  # Watts to kW
            blower_plot_data = ah_energy[blower_rate_col] / 1000.0

        ax.plot(
            ah_plot_times, ac_plot_data, color="magenta", lw=4,
            linestyle="-", label="AHU Cooling Elec."
        )
        ax.plot(
            ah_plot_times, blower_plot_data, color="magenta", lw=4,
            linestyle="--", label="AHU Fan Elec."
        )

    # Plot Boiler energy
    boiler_energy = energy_timeseries[
        energy_timeseries["device_type"] == "boiler"
    ].copy()
    if boiler_energy.empty:
        logger.warning("No boiler energy data to plot.")
    else:
        boiler_energy["start_time"] = pd.to_datetime(boiler_energy["start_time"])
        if boiler_energy["start_time"].dt.tz is None:
             boiler_energy["start_time"] = boiler_energy["start_time"].dt.tz_localize('UTC')
        boiler_plot_times = boiler_energy["start_time"].dt.tz_convert(time_zone_str)

        gas_rate_col = "boiler_natural_gas_heating_energy_rate"
        pump_rate_col = "boiler_pump_electrical_energy_rate"

        if cumulative:
            gas_plot_data = _to_kwh(boiler_energy[gas_rate_col])
            pump_plot_data = _to_kwh(boiler_energy[pump_rate_col])
        else:
            gas_plot_data = boiler_energy[gas_rate_col] / 1000.0
            pump_plot_data = boiler_energy[pump_rate_col] / 1000.0

        ax.plot(
            boiler_plot_times, gas_plot_data, color="lime", lw=4,
            linestyle="-", label="Boiler Gas"
        )
        ax.plot(
            boiler_plot_times, pump_plot_data, color="lime", lw=4,
            linestyle="--", label="Boiler Pump Elec."
        )

    y_axis_label = (
        "HVAC Energy Consumption [kWh]" if cumulative else "HVAC Power [kW]"
    )
    # Use overall min/max times from the full energy_timeseries for x-axis limits
    min_ts = pd.to_datetime(energy_timeseries["start_time"]).min()
    max_ts = pd.to_datetime(energy_timeseries["end_time"]).max()

    if pd.isna(min_ts) or pd.isna(max_ts):
        logger.warning("Min/max timestamps for energy plot are invalid.")
        min_ts, max_ts = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=1), pd.Timestamp.now(tz='UTC')


    self._format_plot(ax, y_axis_label, min_ts, max_ts, time_zone_str)

  def _plot_energy_cost_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone_str: str,
      cumulative: bool = False,
  ) -> None:
    """Plots the energy cost over time.

    Args:
      ax (plt.Axes): The Matplotlib Axes to plot on.
      reward_timeseries (pd.DataFrame): DataFrame with 'electricity_energy_cost'
        and a DatetimeIndex.
      time_zone_str (str): Target time zone for display.
      cumulative (bool): If True, plot cumulative cost, otherwise instantaneous.
    """
    if reward_timeseries.index.tz is not None:
        local_times = reward_timeseries.index.tz_convert(time_zone_str)
    else: # If tz-naive, assume UTC and localize
        local_times = reward_timeseries.index.tz_localize('UTC').tz_convert(time_zone_str)


    cost_data = reward_timeseries["electricity_energy_cost"]
    if cumulative:
      cost_data = cost_data.cumsum()

    ax.plot(
        local_times, cost_data, color="magenta", marker=None, alpha=1,
        lw=2, linestyle="-", label="Electricity Cost"
    )
    self._format_plot(
        ax, "Energy Cost [$]", reward_timeseries.index.min(),
        reward_timeseries.index.max(), time_zone_str
    )

  def _plot_carbon_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone_str: str,
      cumulative: bool = False,
  ) -> None:
    """Plots carbon emissions over time.

    Args:
      ax (plt.Axes): The Matplotlib Axes to plot on.
      reward_timeseries (pd.DataFrame): DataFrame with 'carbon_emitted'
        and a DatetimeIndex.
      time_zone_str (str): Target time zone for display.
      cumulative (bool): If True, plot cumulative emissions.
    """
    if reward_timeseries.index.tz is not None:
        local_times = reward_timeseries.index.tz_convert(time_zone_str)
    else: # If tz-naive, assume UTC and localize
        local_times = reward_timeseries.index.tz_localize('UTC').tz_convert(time_zone_str)

    carbon_data = reward_timeseries["carbon_emitted"]
    if cumulative:
      carbon_data = carbon_data.cumsum()

    ax.plot(
        local_times, carbon_data, color="white", marker=None, alpha=1,
        lw=4, linestyle="-", label="Carbon Emissions"
    )
    self._format_plot(
        ax, "Carbon Emission [kg CO2eq]", reward_timeseries.index.min(),
        reward_timeseries.index.max(), time_zone_str
    )

  def _plot_occupancy_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone_str: str
  ) -> None:
    """Plots building occupancy over time.

    Args:
      ax (plt.Axes): The Matplotlib Axes to plot on.
      reward_timeseries (pd.DataFrame): DataFrame with an 'occupancy' column
        and a DatetimeIndex.
      time_zone_str (str): Target time zone for display.
    """
    if reward_timeseries.index.tz is not None:
        local_times = reward_timeseries.index.tz_convert(time_zone_str)
    else: # If tz-naive, assume UTC and localize
        local_times = reward_timeseries.index.tz_localize('UTC').tz_convert(time_zone_str)

    ax.plot(
        local_times, reward_timeseries["occupancy"], color="cyan",
        marker=None, alpha=1, lw=2, linestyle="-", label="Num. Occupants"
    )
    self._format_plot(
        ax, "Occupancy", reward_timeseries.index.min(),
        reward_timeseries.index.max(), time_zone_str
    )

  def _plot_temperature_timeline(
      self,
      ax: plt.Axes,
      zone_timeseries: pd.DataFrame,
      outside_air_temp_series: pd.Series, # Series with DatetimeIndex
      time_zone_str: str,
  ) -> None:
    """Plots zone temperatures, setpoints, and outside air temperature.

    Args:
      ax (plt.Axes): The Matplotlib Axes to plot on.
      zone_timeseries (pd.DataFrame): DataFrame with zone temperature data,
        including 'start_time', 'zone', 'zone_air_temperature',
        'heating_setpoint_temperature', 'cooling_setpoint_temperature'.
      outside_air_temp_series (pd.Series): Series of outside air temperatures
        with a DatetimeIndex.
      time_zone_str (str): Target time zone for display.
    """
    if zone_timeseries.empty:
        logger.warning("Zone timeseries is empty, cannot plot temperatures.")
        # Format with current time if no data
        now_utc = pd.Timestamp.now(tz='UTC')
        self._format_plot(ax, "Temperature [C]", now_utc - pd.Timedelta(hours=1), now_utc, time_zone_str)
        return

    # Ensure 'start_time' is datetime and timezone-aware (assume UTC if naive)
    zone_timeseries['start_time'] = pd.to_datetime(zone_timeseries['start_time'])
    if zone_timeseries['start_time'].dt.tz is None:
        zone_timeseries['start_time'] = zone_timeseries['start_time'].dt.tz_localize('UTC')

    # Pivot to get temperatures per zone over time
    zone_temps_pivot = zone_timeseries.pivot_table(
        index="start_time", columns="zone", values="zone_air_temperature"
    )
    # Calculate statistics across zones
    zone_temp_stats = pd.DataFrame({
        "min_temp": zone_temps_pivot.min(axis=1),
        "q25_temp": zone_temps_pivot.quantile(q=0.25, axis=1),
        "median_temp": zone_temps_pivot.median(axis=1),
        "q75_temp": zone_temps_pivot.quantile(q=0.75, axis=1),
        "max_temp": zone_temps_pivot.max(axis=1),
    })
    # Convert index to local time for plotting
    plot_times_local = zone_temp_stats.index.tz_convert(time_zone_str)

    # Heating/Cooling setpoints (min heating, max cooling across zones)
    heating_sp = zone_timeseries.pivot_table(
        index="start_time", columns="zone", values="heating_setpoint_temperature"
    ).min(axis=1)
    cooling_sp = zone_timeseries.pivot_table(
        index="start_time", columns="zone", values="cooling_setpoint_temperature"
    ).max(axis=1)

    # Plot setpoints
    ax.plot(
        heating_sp.index.tz_convert(time_zone_str),
        heating_sp - self.KELVIN_TO_CELSIUS, color="yellow", lw=1, label="Heat SP (min)"
    )
    ax.plot(
        cooling_sp.index.tz_convert(time_zone_str),
        cooling_sp - self.KELVIN_TO_CELSIUS, color="yellow", lw=1, label="Cool SP (max)"
    )

    # Plot temperature ranges (min-max, interquartile)
    ax.fill_between(
        plot_times_local,
        zone_temp_stats["min_temp"] - self.KELVIN_TO_CELSIUS,
        zone_temp_stats["max_temp"] - self.KELVIN_TO_CELSIUS,
        facecolor="green", alpha=0.4, label="Zone Temp Min-Max"
    )
    ax.fill_between(
        plot_times_local,
        zone_temp_stats["q25_temp"] - self.KELVIN_TO_CELSIUS,
        zone_temp_stats["q75_temp"] - self.KELVIN_TO_CELSIUS,
        facecolor="green", alpha=0.8, label="Zone Temp IQR"
    )
    # Plot median zone temperature
    ax.plot(
        plot_times_local,
        zone_temp_stats["median_temp"] - self.KELVIN_TO_CELSIUS,
        color="white", lw=3, alpha=1.0, label="Zone Temp Median"
    )

    # Plot outside air temperature
    if not outside_air_temp_series.empty:
        if outside_air_temp_series.index.tz is None:
            outside_air_temp_series.index = outside_air_temp_series.index.tz_localize('UTC')
        oat_plot_times = outside_air_temp_series.index.tz_convert(time_zone_str)
        ax.plot(
            oat_plot_times,
            outside_air_temp_series - self.KELVIN_TO_CELSIUS,
            color="magenta", lw=3, alpha=1.0, label="Outside Air Temp"
        )

    self._format_plot(
        ax, "Temperature [C]", zone_temp_stats.index.min(),
        zone_temp_stats.index.max(), time_zone_str
    )

  def _plot_action_timeline(
      self,
      ax: plt.Axes,
      action_timeseries: pd.DataFrame,
      action_tuple: tuple[str, str],
      time_zone_str: str,
  ) -> None:
    """Plots a specific agent action's timeline.

    Args:
      ax (plt.Axes): The Matplotlib Axes to plot on.
      action_timeseries (pd.DataFrame): DataFrame of action data, with
        'device_id', 'setpoint_name', 'timestamp', 'setpoint_value'.
      action_tuple (tuple[str, str]): A (device_id, setpoint_name) tuple
        identifying the action to plot.
      time_zone_str (str): Target time zone for display.
    """
    dev_id, sp_name = action_tuple
    single_action_df = action_timeseries[
        (action_timeseries["device_id"] == dev_id) &
        (action_timeseries["setpoint_name"] == sp_name)
    ].sort_values(by="timestamp")

    if single_action_df.empty:
        logger.warning("No data for action: %s", action_tuple)
        # Format with current time if no data
        now_utc = pd.Timestamp.now(tz='UTC')
        self._format_plot(ax, f"Action: {sp_name}", now_utc - pd.Timedelta(hours=1), now_utc, time_zone_str)
        return

    # Ensure 'timestamp' is datetime and timezone-aware
    single_action_df["timestamp"] = pd.to_datetime(single_action_df["timestamp"])
    if single_action_df["timestamp"].dt.tz is None:
        single_action_df["timestamp"] = single_action_df["timestamp"].dt.tz_localize('UTC')
    plot_times_local = single_action_df["timestamp"].dt.tz_convert(time_zone_str)


    plot_values = single_action_df["setpoint_value"]
    y_label = f"Action: {sp_name}"
    # Convert to Celsius if it's a known temperature setpoint
    if sp_name in [
        "supply_water_setpoint", "supply_air_heating_temperature_setpoint"
    ]:
      plot_values = plot_values - self.KELVIN_TO_CELSIUS
      y_label += " [C]"

    ax.plot(
        plot_times_local, plot_values, color="lime", marker=None, alpha=1,
        lw=4, linestyle="-", label=sp_name
    )
    self._format_plot(
        ax, y_label, single_action_df["timestamp"].min(),
        single_action_df["timestamp"].max(), time_zone_str
    )

  def _plot_timeseries_charts(
      self,
      reader: data_reader.EpisodeDataReader,
      time_zone_str: str,
      step_count: int
  ) -> None:
    """Generates and saves a multi-panel plot of episode time-series data.

    Args:
      reader (data_reader.EpisodeDataReader): Reader for accessing episode data.
      time_zone_str (str): Target time zone for display.
      step_count (int): Current step count, used for naming the saved file.
    """
    min_time = pd.Timestamp.min.tz_localize('UTC')
    max_time = pd.Timestamp.max.tz_localize('UTC')

    obs_responses = reader.read_observation_responses(min_time, max_time)
    action_responses = reader.read_action_responses(min_time, max_time)
    reward_infos = reader.read_reward_infos(min_time, max_time)
    reward_responses = reader.read_reward_responses(min_time, max_time)

    if not reward_infos or not reward_responses:
      logger.info("No reward data available for plotting.")
      return

    action_ts = get_action_timeseries(action_responses)
    # Get unique (device_id, setpoint_name) tuples for plotting individual actions
    action_tuples = list(
        action_ts[["device_id", "setpoint_name"]].drop_duplicates().itertuples(index=False, name=None)
    ) if not action_ts.empty else []


    reward_ts = get_reward_timeseries(reward_infos, reward_responses, time_zone_str)
    if reward_ts.empty:
        logger.info("Reward timeseries is empty after processing.")
        return # Cannot proceed without reward timestamps for other plots

    oat_ts = get_outside_air_temperature_timeseries(obs_responses, time_zone_str)
    zone_ts = get_zone_timeseries(reward_infos, time_zone_str) # Uses RewardInfo for zone data
    energy_ts = get_energy_timeseries(reward_infos, time_zone_str) # Uses RewardInfo

    num_action_plots = len(action_tuples)
    num_rows = 6 + num_action_plots
    height_ratios = [1] * num_rows

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=1,
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False, # Always return 2D array for axes
        figsize=(24, 6 * num_rows) # Adjust height based on number of plots
    )
    axes = axes.flatten() # Flatten to 1D array for easier indexing

    # Plotting each panel
    self._plot_reward_timeline(axes[0], reward_ts, time_zone_str)
    if not energy_ts.empty:
        self._plot_energy_timeline(axes[1], energy_ts, time_zone_str, cumulative=True)
    else:
        logger.warning("Energy timeseries is empty. Skipping energy plot.")
    self._plot_energy_cost_timeline(axes[2], reward_ts, time_zone_str, cumulative=True)
    self._plot_carbon_timeline(axes[3], reward_ts, time_zone_str, cumulative=True)
    self._plot_occupancy_timeline(axes[4], reward_ts, time_zone_str)
    self._plot_temperature_timeline(axes[5], zone_ts, oat_ts, time_zone_str)

    for i, act_tuple in enumerate(action_tuples):
      if not action_ts.empty:
          self._plot_action_timeline(axes[6 + i], action_ts, act_tuple, time_zone_str)
      else:
          logger.warning("Action timeseries is empty. Skipping action plot for %s.", act_tuple)


    fig_path = os.path.join(self._save_path, f"timeseries_step_{step_count}.png")
    try:
        fig.savefig(fig_path, bbox_inches="tight", dpi=100)
        logger.info("Saved timeseries plot to %s", fig_path)
    except Exception as e: # pylint: disable=broad-except
        logger.error("Failed to save timeseries plot: %s", e)
    finally:
        plt.close(fig) # Ensure figure is closed to free memory

  def _render_env(
      self, current_env: smart_control_environment.Environment, step_count: int
  ) -> None:
    """Renders the current state of the environment and saves it as an image.

    This method assumes the environment has a specific structure to access
    building layout, temperatures, and heat inputs for rendering.

    Args:
      current_env (smart_control_environment.Environment): The smart building
        control environment instance.
      step_count (int): Current step count, used for naming the saved file.
    """
    try:
      # These attributes are specific to the custom Environment structure
      building_sim = current_env.building.simulator # type: ignore[attr-defined]
      building_layout = building_sim.building.floor_plan
      temps = building_sim.building.temp
      input_q = building_sim.building.input_q
    except AttributeError as e:
      logger.error("Failed to get data for rendering from environment: %s", e)
      return

    renderer = building_renderer.BuildingRenderer(building_layout, 1)
    image = renderer.render(
        temps, cmap="bwr", vmin=285, vmax=305, colorbar=False,
        input_q=input_q, diff_range=0.5, diff_size=1
    ).convert("RGB")

    timestamp_str = current_env.current_simulation_timestamp.strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(
        self._save_path, f"env_render_step_{step_count}_{timestamp_str}.png"
    )
    try:
        image.save(img_path)
        logger.info("Saved environment render to %s", img_path)
    except Exception as e: # pylint: disable=broad-except
        logger.error("Failed to save environment render: %s", e)


  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Processes a trajectory and triggers rendering/plotting if interval met.

    Args:
      trajectory (trajectory_lib.Trajectory): The trajectory data from the
        agent's step.
    """
    reward_value = trajectory.reward
    if hasattr(reward_value, "numpy"): # TF tensor
      reward_value = reward_value.numpy()
    self._cumulative_reward += float(reward_value)
    self._counter += 1

    if self._start_time is None:
      self._start_time = pd.Timestamp.now()

    if self._counter % self._render_interval_steps == 0 and self._environment:
      logger.info("RenderingObserver: Interval reached at step %d.", self._counter)
      current_py_env = self._environment.pyenv # type: ignore[attr-defined]
      # Assuming it's a batched env, get the first one.
      # This might need to be more robust depending on env structure.
      if isinstance(current_py_env, py_environment.PyEnvironment) and hasattr(current_py_env, 'envs'):
          actual_env = current_py_env.envs[0]
      else: # Fallback or direct use if not batched as expected
          actual_env = current_py_env


      if not isinstance(actual_env, smart_control_environment.Environment):
          logger.error(
              "Underlying environment is not of expected type "
              "smart_control.environment.Environment. Cannot render/plot."
          )
          return

      execution_time_td = pd.Timestamp.now() - self._start_time
      mean_exec_time_s = execution_time_td.total_seconds() / self._counter
      logger.info(
          "Step %d: Cum. Reward = %.2f, Mean Step Exec Time = %.2fs",
          self._counter, float(self._cumulative_reward), mean_exec_time_s
      )

      if actual_env.metrics_path:
        logger.info("Plotting timeseries charts from metrics path: %s", actual_env.metrics_path)
        try:
            reader = get_latest_episode_reader(actual_env.metrics_path)
            if reader:
                self._plot_timeseries_charts(reader, self._time_zone, self._counter)
            else:
                logger.warning("No episode data reader found at path: %s", actual_env.metrics_path)
        except Exception as e: # pylint: disable=broad-except
            logger.error("Error during timeseries plotting: %s", e, exc_info=True)


      if self._render_fn:
          self._render_fn() # Call custom render function
      else: # Default rendering
          self._render_env(actual_env, self._counter)


  def reset(self) -> None:
    """Resets the observer's internal counters and timers.
    This is typically called at the start of a new episode.
    """
    self._counter = 0
    self._cumulative_reward = 0.0
    self._start_time = None
    logger.info("RenderingObserver has been reset.")
