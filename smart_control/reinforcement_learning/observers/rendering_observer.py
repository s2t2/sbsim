"""Provides an RL observer for rendering environment states and plotting metrics.

This module defines the `RenderingObserver` class, which is designed to
visualize the state of an RL environment and the agent's performance. It can
periodically render the environment (e.g., as an image of a building's thermal
distribution) and generate various time-series plots of collected metrics from
the episode data, such as rewards, energy consumption, temperatures, and agent
actions. All visualizations are saved to disk.
"""

import logging
import os
from typing import Callable, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import pytz # For timezone handling in plots
from tf_agents.environments import py_environment # For type hinting env
from tf_agents.trajectories import trajectory as trajectory_lib


from smart_control.environment import environment as smart_control_env # Specific environment type
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
from smart_control.utils import building_renderer # For the default rendering
from smart_control.utils import reader_lib # For type hinting reader

logger = logging.getLogger(__name__)


class RenderingObserver(Observer):
  """Renders environment states and plots metrics at specified intervals.

  This observer performs two main functions:
  1.  Periodically renders the current state of the environment using either a
      provided `render_fn` or a default building renderer (`_render_env`).
  2.  Periodically generates and saves time-series plots of various metrics
      (e.g., reward, energy, temperature, actions) using either a provided
      `plot_fn` or a default plotting mechanism (`_plot_timeseries_charts`)
      that reads data from the environment's metrics log.

  Visualizations are saved to a specified directory.
  """

  # Class constant for temperature conversion
  KELVIN_TO_CELSIUS = _KELVIN_TO_CELSIUS

  def __init__(
      self,
      render_interval_steps: int = 10,
      env: Optional[py_environment.PyEnvironment] = None,
      render_fn: Optional[Callable[[smart_control_env.Environment, int], None]] = None,
      plot_fn: Optional[Callable[[reader_lib.MetricsReader, str, int], None]] = None,
      clear_output_before_render: bool = True, # TODO(user): Implement or remove
      time_zone: str = DEFAULT_TIME_ZONE,
      save_path: str = RENDERS_PATH,
  ):
    """Initializes the RenderingObserver.

    Args:
      render_interval_steps: The number of environment steps between each
        rendering and plotting action.
      env: The `tf_agents.environments.PyEnvironment` to observe. This is
        expected to wrap a `smart_control.environment.Environment` to access
        simulation timestamps, metrics paths, and rendering capabilities.
      render_fn: An optional custom function to render the environment's state.
        It should accept the `smart_control.environment.Environment` instance
        and the current step count as arguments. If `None`, `_render_env` is used.
      plot_fn: An optional custom function to generate plots from episode data.
        It should accept a `reader_lib.MetricsReader` instance, the timezone
        string, and the current step count. If `None`,
        `_plot_timeseries_charts` is used.
      clear_output_before_render: Intended to clear previous output before
        rendering, especially in notebook environments. (Currently not
        implemented in the `__call__` method).
      time_zone: The timezone string (e.g., "America/Los_Angeles", "UTC")
        used for displaying timestamps in plots.
      save_path: The directory path where rendered images and plots will be
        saved. This directory will be created if it doesn't exist.
    """
    self._counter = 0
    self._render_interval_steps = render_interval_steps
    self._environment = env
    self._render_fn = render_fn if render_fn else self._render_env
    self._plot_fn = plot_fn if plot_fn else self._plot_timeseries_charts
    self._clear_output_before_render = clear_output_before_render # Not currently used
    self._time_zone = time_zone
    self._cumulative_reward = 0.0
    self._start_time: Optional[pd.Timestamp] = None
    self._save_path = save_path
    self._num_timesteps_in_episode = 0 # Default if env not available

    # Create save directory if it doesn't exist
    os.makedirs(self._save_path, exist_ok=True)

    if self._environment is not None and hasattr(self._environment, 'pyenv'):
      # Attempt to access the underlying smart_control.environment.Environment
      # This assumes the pyenv is wrapping a single environment or a list.
      inner_env = self._environment.pyenv
      if isinstance(inner_env, list) and inner_env: # Handle batched envs
          inner_env = inner_env[0]
      elif hasattr(inner_env, 'envs') and inner_env.envs: # Handle TFPyEnvironment
          inner_env = inner_env.envs[0]

      if hasattr(inner_env, '_num_timesteps_in_episode'):
        self._num_timesteps_in_episode = inner_env._num_timesteps_in_episode # pylint: disable=protected-access

  def _format_plot(
      self,
      ax: plt.Axes,
      ylabel: str,
      start_time: pd.Timestamp,
      end_time: pd.Timestamp,
      time_zone: str
  ) -> None:
    """Applies common formatting to a matplotlib Axes object for time-series plots.

    Args:
      ax: The `matplotlib.axes.Axes` object to format.
      ylabel: The label for the y-axis.
      start_time: The minimum timestamp for the x-axis limit.
      end_time: The maximum timestamp for the x-axis limit.
      time_zone: The timezone string for formatting x-axis date labels.
    """
    ax.set_facecolor('black')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter('%a %m/%d %H:%M', tz=pytz.timezone(time_zone))
    )
    ax.grid(color='gray', linestyle='-', linewidth=1.0)
    ax.set_ylabel(ylabel, color='blue', fontsize=12) # Consider making color configurable
    ax.set_xlim(left=start_time, right=end_time)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if ax.has_data(): # Only show legend if there are plotted elements
        ax.legend(prop={'size': 10})

  def _plot_reward_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone: str
  ) -> None:
    """Plots the cumulative agent reward over time.

    Args:
      ax: The `matplotlib.axes.Axes` to plot on.
      reward_timeseries: A DataFrame with a 'cumulative_reward' column and
        a DatetimeIndex.
      time_zone: The timezone string for x-axis formatting.
    """
    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]

    ax.plot(
        local_times,
        reward_timeseries['cumulative_reward'],
        color='royalblue',
        marker=None,
        alpha=1.0, # Changed to float
        lw=6,
        linestyle='-',
        label='Cumulative Reward',
    )
    self._format_plot(
        ax,
        'Agent Reward',
        reward_timeseries.index.min(), # pytype: disable=attribute-error
        reward_timeseries.index.max(), # pytype: disable=attribute-error
        time_zone,
    )

  def _plot_energy_timeline(
      self,
      ax: plt.Axes,
      energy_timeseries: pd.DataFrame,
      time_zone: str,
      cumulative: bool = False
  ) -> None:
    """Plots HVAC energy consumption (power or cumulative kWh) over time.

    Distinguishes between Air Handling Unit (AHU) electricity (A/C and fan)
    and Boiler (BLR) energy (gas and pump electricity).

    Args:
      ax: The `matplotlib.axes.Axes` to plot on.
      energy_timeseries: DataFrame containing energy data, expected to have
        columns like 'device_type', 'air_handler_air_conditioner_energy_rate',
        'boiler_natural_gas_heating_energy_rate', etc., and 'start_time'.
      time_zone: The timezone string for x-axis formatting.
      cumulative: If True, plots cumulative energy in kWh. Otherwise, plots
        power in kW.
    """
    def _to_kwh(
        energy_rate_series: pd.Series, # Changed to Series for type hint
        step_interval: pd.Timedelta = pd.Timedelta(5, unit='minutes')
    ) -> pd.Series: # Return type is pd.Series
      """Converts energy rate (Watts) to cumulative kWh."""
      kw_power = energy_rate_series / 1000.0 # Watts to kW
      # Energy in kWh for the interval = kW * (interval hours)
      kwh_interval = kw_power * (step_interval.total_seconds() / 3600.0)
      return kwh_interval.cumsum()

    # Filter for AHU data
    ahu_timeseries = energy_timeseries[energy_timeseries['device_type'] == 'air_handler']
    if not ahu_timeseries.empty:
      if cumulative:
        feature_timeseries_ac = _to_kwh(ahu_timeseries['air_handler_air_conditioner_energy_rate'])
        feature_timeseries_blower = _to_kwh(ahu_timeseries['air_handler_blower_electrical_energy_rate'])
      else:
        feature_timeseries_ac = ahu_timeseries['air_handler_air_conditioner_energy_rate'] / 1000.0
        feature_timeseries_blower = ahu_timeseries['air_handler_blower_electrical_energy_rate'] / 1000.0

      ax.plot(
          ahu_timeseries['start_time'], feature_timeseries_ac, color='magenta',
          marker=None, alpha=1.0, lw=4, linestyle='-', label='AHU A/C'
      )
      ax.plot(
          ahu_timeseries['start_time'], feature_timeseries_blower, color='magenta',
          marker=None, alpha=1.0, lw=4, linestyle='--', label='AHU Fan'
      )

    # Filter for Boiler data
    boiler_timeseries = energy_timeseries[energy_timeseries['device_type'] == 'boiler']
    if not boiler_timeseries.empty:
      if cumulative:
        feature_timeseries_gas = _to_kwh(boiler_timeseries['boiler_natural_gas_heating_energy_rate'])
        feature_timeseries_pump = _to_kwh(boiler_timeseries['boiler_pump_electrical_energy_rate'])
      else:
        feature_timeseries_gas = boiler_timeseries['boiler_natural_gas_heating_energy_rate'] / 1000.0
        feature_timeseries_pump = boiler_timeseries['boiler_pump_electrical_energy_rate'] / 1000.0

      ax.plot(
          boiler_timeseries['start_time'], feature_timeseries_gas, color='lime',
          marker=None, alpha=1.0, lw=4, linestyle='-', label='Boiler Gas'
      )
      ax.plot(
          boiler_timeseries['start_time'], feature_timeseries_pump, color='lime',
          marker=None, alpha=1.0, lw=4, linestyle='--', label='Boiler Pump'
      )

    y_label = 'HVAC Energy [kWh]' if cumulative else 'HVAC Power [kW]'
    # Use overall min/max times for x-axis if data exists
    min_time = energy_timeseries['start_time'].min() if not energy_timeseries.empty else pd.Timestamp.now()
    max_time = energy_timeseries['end_time'].max() if not energy_timeseries.empty else pd.Timestamp.now()

    self._format_plot(ax, y_label, min_time, max_time, time_zone)

  def _plot_energy_cost_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone: str,
      cumulative: bool = False,
  ) -> None:
    """Plots the energy cost (specifically electricity) over time.

    Args:
      ax: The `matplotlib.axes.Axes` to plot on.
      reward_timeseries: DataFrame with 'electricity_energy_cost' column and
        DatetimeIndex.
      time_zone: The timezone string for x-axis formatting.
      cumulative: If True, plots cumulative cost. Otherwise, plots cost per step.
    """
    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]

    if cumulative:
      feature_timeseries_cost = reward_timeseries['electricity_energy_cost'].cumsum()
    else:
      feature_timeseries_cost = reward_timeseries['electricity_energy_cost']

    ax.plot(
        local_times, feature_timeseries_cost, color='magenta', marker=None,
        alpha=1.0, lw=2, linestyle='-', label='Electricity Cost'
    )

    self._format_plot(
        ax,
        'Energy Cost [$]',
        reward_timeseries.index.min(), # pytype: disable=attribute-error
        reward_timeseries.index.max(), # pytype: disable=attribute-error
        time_zone,
    )

  def _plot_carbon_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone: str,
      cumulative: bool = False
  ) -> None:
    """Plots carbon emissions over time.

    Args:
      ax: The `matplotlib.axes.Axes` to plot on.
      reward_timeseries: DataFrame with 'carbon_emitted' column and DatetimeIndex.
      time_zone: The timezone string for x-axis formatting.
      cumulative: If True, plots cumulative carbon emissions. Otherwise, plots
        emissions per step.
    """
    if cumulative:
      feature_timeseries_carbon = reward_timeseries['carbon_emitted'].cumsum()
    else:
      feature_timeseries_carbon = reward_timeseries['carbon_emitted']

    ax.plot(
        reward_timeseries.index, feature_timeseries_carbon, color='white',
        marker=None, alpha=1.0, lw=4, linestyle='-', label='Carbon Emissions'
    )

    self._format_plot(
        ax,
        'Carbon Emission [kg]',
        reward_timeseries.index.min(), # pytype: disable=attribute-error
        reward_timeseries.index.max(), # pytype: disable=attribute-error
        time_zone,
    )

  def _plot_occupancy_timeline(
      self,
      ax: plt.Axes,
      reward_timeseries: pd.DataFrame,
      time_zone: str
  ) -> None:
    """Plots the number of occupants over time.

    Args:
      ax: The `matplotlib.axes.Axes` to plot on.
      reward_timeseries: DataFrame with 'occupancy' column and DatetimeIndex.
      time_zone: The timezone string for x-axis formatting.
    """
    local_times = [ts.tz_convert(time_zone) for ts in reward_timeseries.index]

    ax.plot(
        local_times, reward_timeseries['occupancy'], color='cyan', marker=None,
        alpha=1.0, lw=2, linestyle='-', label='Number of Occupants'
    )

    self._format_plot(
        ax,
        'Occupancy',
        reward_timeseries.index.min(), # pytype: disable=attribute-error
        reward_timeseries.index.max(), # pytype: disable=attribute-error
        time_zone,
    )

  def _plot_temperature_timeline(
      self,
      ax: plt.Axes,
      zone_timeseries: pd.DataFrame,
      outside_air_temperature_timeseries: pd.Series, # Changed for clarity
      time_zone: str
  ) -> None:
    """Plots zone temperature statistics and outside air temperature over time.

    Displays min, max, median, and interquartile range (IQR) of zone temperatures,
    along with heating/cooling setpoints and outside air temperature.

    Args:
      ax: The `matplotlib.axes.Axes` to plot on.
      zone_timeseries: DataFrame with zone temperature data, expected to have
        columns 'start_time', 'zone', 'zone_air_temperature',
        'heating_setpoint_temperature', 'cooling_setpoint_temperature'.
      outside_air_temperature_timeseries: Series of outside air temperatures
        with a DatetimeIndex.
      time_zone: The timezone string for x-axis formatting.
    """
    if zone_timeseries.empty:
        logger.warning("Zone timeseries data is empty. Skipping temperature plot.")
        self._format_plot(ax, 'Temperature [C]', pd.Timestamp.now(tz=time_zone), pd.Timestamp.now(tz=time_zone) + pd.Timedelta(hours=1), time_zone)
        return

    zone_temps = pd.pivot_table(
        zone_timeseries,
        index='start_time', # Pivot on 'start_time'
        columns='zone',
        values='zone_air_temperature'
    ).sort_index()

    if zone_temps.empty:
        logger.warning("Pivoted zone_temps data is empty. Skipping temperature plot details.")
        self._format_plot(ax, 'Temperature [C]', pd.Timestamp.now(tz=time_zone), pd.Timestamp.now(tz=time_zone) + pd.Timedelta(hours=1), time_zone)
        return

    zone_temp_stats = pd.DataFrame({
        'min_temp': zone_temps.min(axis=1),
        'q25_temp': zone_temps.quantile(q=0.25, axis=1),
        'median_temp': zone_temps.median(axis=1),
        'q75_temp': zone_temps.quantile(q=0.75, axis=1),
        'max_temp': zone_temps.max(axis=1),
    })

    zone_heating_setpoints = (
        pd.pivot_table(
            zone_timeseries, index='start_time', columns='zone',
            values='heating_setpoint_temperature'
        ).sort_index().min(axis=1) # Get the minimum heating setpoint across zones
    )
    zone_cooling_setpoints = (
        pd.pivot_table(
            zone_timeseries, index='start_time', columns='zone',
            values='cooling_setpoint_temperature'
        ).sort_index().max(axis=1) # Get the maximum cooling setpoint across zones
    )

    # Plotting setpoint bands
    ax.plot(
        zone_cooling_setpoints.index, zone_cooling_setpoints - self.KELVIN_TO_CELSIUS,
        color='yellow', lw=1, label='Cooling Setpoint (Max)'
    )
    ax.plot(
        zone_heating_setpoints.index, zone_heating_setpoints - self.KELVIN_TO_CELSIUS,
        color='orange', lw=1, label='Heating Setpoint (Min)' # Changed color for distinction
    )

    # Plotting temperature ranges (min-max and IQR)
    ax.fill_between(
        zone_temp_stats.index,
        zone_temp_stats['min_temp'] - self.KELVIN_TO_CELSIUS,
        zone_temp_stats['max_temp'] - self.KELVIN_TO_CELSIUS,
        facecolor='green', alpha=0.3, label='Zone Temp Range (Min-Max)'
    )
    ax.fill_between(
        zone_temp_stats.index,
        zone_temp_stats['q25_temp'] - self.KELVIN_TO_CELSIUS,
        zone_temp_stats['q75_temp'] - self.KELVIN_TO_CELSIUS,
        facecolor='green', alpha=0.6, label='Zone Temp IQR'
    )
    ax.plot(
        zone_temp_stats.index, zone_temp_stats['median_temp'] - self.KELVIN_TO_CELSIUS,
        color='white', lw=2, alpha=1.0, label='Zone Median Temp' # Reduced lw for clarity
    )

    # Plotting outside air temperature
    if not outside_air_temperature_timeseries.empty:
      ax.plot(
          outside_air_temperature_timeseries.index,
          outside_air_temperature_timeseries - self.KELVIN_TO_CELSIUS,
          color='magenta', lw=2, alpha=1.0, label='Outside Air Temp' # Reduced lw
      )

    self._format_plot(
        ax,
        'Temperature [C]',
        zone_temp_stats.index.min(),
        zone_temp_stats.index.max(),
        time_zone,
    )

  def _plot_action_timeline(
      self,
      ax: plt.Axes,
      action_timeseries: pd.DataFrame,
      action_tuple: tuple[str, str],
      time_zone: str
  ) -> None:
    """Plots a specific agent action (setpoint) over time.

    Args:
      ax: The `matplotlib.axes.Axes` to plot on.
      action_timeseries: DataFrame containing action data, expected to have
        columns 'device_id', 'setpoint_name', 'timestamp', 'setpoint_value'.
      action_tuple: A tuple `(device_id, setpoint_name)` specifying which
        action to plot.
      time_zone: The timezone string for x-axis formatting.
    """
    single_action_timeseries = action_timeseries[
        (action_timeseries['device_id'] == action_tuple[0]) &
        (action_timeseries['setpoint_name'] == action_tuple[1])
    ].sort_values(by='timestamp')

    if single_action_timeseries.empty:
        logger.warning("No data for action: %s. Skipping plot.", action_tuple)
        self._format_plot(ax, f'Action: {action_tuple[1]}', pd.Timestamp.now(tz=time_zone), pd.Timestamp.now(tz=time_zone) + pd.Timedelta(hours=1), time_zone)
        return

    # Convert to Celsius if applicable
    plot_values = single_action_timeseries['setpoint_value']
    if action_tuple[1] in ['supply_water_setpoint', 'supply_air_heating_temperature_setpoint']:
      plot_values = plot_values - self.KELVIN_TO_CELSIUS

    ax.plot(
        single_action_timeseries['timestamp'], plot_values, color='lime',
        marker=None, alpha=1.0, lw=4, linestyle='-', label=action_tuple[1]
    )

    self._format_plot(
        ax,
        f'Action: {action_tuple[1]}', # Y-label indicates the action
        single_action_timeseries['timestamp'].min(),
        single_action_timeseries['timestamp'].max(),
        time_zone,
    )

  def _plot_timeseries_charts(
      self,
      reader: reader_lib.MetricsReader,
      time_zone: str,
      step_count: int
  ) -> None:
    """Generates and saves a multi-panel plot of various time-series metrics.

    This method reads data for observations, actions, rewards, etc., from the
    provided `reader`, processes it into time-series, and then calls
    individual plotting methods (e.g., `_plot_reward_timeline`) to create
    a figure with multiple subplots. The resulting figure is saved to a PNG file.

    Args:
      reader: A `reader_lib.MetricsReader` instance for accessing logged
        episode data.
      time_zone: The timezone string for formatting plot timestamps.
      step_count: The current environment step count, used for naming the
        output file.
    """
    # Read data using the provided reader
    # Ensure pd.Timestamp.min and pd.Timestamp.max are timezone-naive or match reader's expectation
    min_ts, max_ts = pd.Timestamp.min, pd.Timestamp.max

    observation_responses = reader.read_observation_responses(min_ts, max_ts)
    action_responses = reader.read_action_responses(min_ts, max_ts)
    reward_infos = reader.read_reward_infos(min_ts, max_ts)
    reward_responses = reader.read_reward_responses(min_ts, max_ts)

    if not reward_infos or not reward_responses:
      logger.info('No reward data available for plotting at step %d.', step_count)
      return

    # Process data into timeseries
    action_timeseries = get_action_timeseries(action_responses)
    action_tuples = list(set([
        (row['device_id'], row['setpoint_name'])
        for _, row in action_timeseries.iterrows()
    ])) if not action_timeseries.empty else []

    reward_timeseries = get_reward_timeseries(reward_infos, reward_responses, time_zone).sort_index()
    energy_timeseries = get_energy_timeseries(reward_infos, time_zone)
    outside_air_temp_ts = get_outside_air_temperature_timeseries(observation_responses, time_zone)
    zone_timeseries = get_zone_timeseries(reward_infos, time_zone)

    # Create figure and axes
    num_subplots = 6 + len(action_tuples)
    fig, axes = plt.subplots(
        nrows=num_subplots, ncols=1,
        gridspec_kw={'height_ratios': [1] * num_subplots},
        squeeze=False # Always return 2D array for axes
    )
    fig.set_size_inches(24, 5 * num_subplots) # Adjust height based on num_subplots

    # Plot individual timelines
    self._plot_reward_timeline(axes[0, 0], reward_timeseries, time_zone)
    self._plot_energy_timeline(axes[1, 0], energy_timeseries, time_zone, cumulative=True)
    self._plot_energy_cost_timeline(axes[2, 0], reward_timeseries, time_zone, cumulative=True)
    self._plot_carbon_timeline(axes[3, 0], reward_timeseries, time_zone, cumulative=True)
    self._plot_occupancy_timeline(axes[4, 0], reward_timeseries, time_zone)
    self._plot_temperature_timeline(axes[5, 0], zone_timeseries, outside_air_temp_ts, time_zone)

    for i, action_tuple in enumerate(action_tuples):
      self._plot_action_timeline(axes[6 + i, 0], action_timeseries, action_tuple, time_zone)

    # Save figure
    fig_path = os.path.join(self._save_path, f'timeseries_step_{step_count}.png')
    try:
      fig.savefig(fig_path, bbox_inches='tight', dpi=100)
      logger.info('Saved timeseries plot to %s', fig_path)
    except Exception as e: # pylint: disable=broad-except
      logger.error("Failed to save timeseries plot: %s", e)
    finally:
      plt.close(fig) # Ensure figure is closed

  def _render_env(
      self,
      env_to_render: smart_control_env.Environment,
      step_count: int
  ) -> None:
    """Renders the environment's current state and saves it as an image.

    This default rendering function visualizes the building's floor plan,
    temperature distribution, and heat inputs using `BuildingRenderer`.
    The rendered image is saved to a PNG file.

    Args:
      env_to_render: The `smart_control.environment.Environment` instance to render.
      step_count: The current environment step count, used for naming the output file.
    """
    # pylint: disable=protected-access
    # Accessing internal simulator details; consider adding public interfaces if possible
    if not hasattr(env_to_render, 'building') or \
       not hasattr(env_to_render.building, 'simulator') or \
       not hasattr(env_to_render.building.simulator, 'building'):
        logger.warning("Environment does not have expected structure for rendering.")
        return

    sim_building = env_to_render.building.simulator.building
    building_layout = sim_building.floor_plan
    temps = sim_building.temp
    input_q = sim_building.input_q
    # pylint: enable=protected-access

    renderer = building_renderer.BuildingRenderer(building_layout, 1)
    image = renderer.render(
        temps, cmap='bwr', vmin=285, vmax=305, colorbar=False,
        input_q=input_q, diff_range=0.5, diff_size=1
    ).convert('RGB')

    timestamp_str = env_to_render.current_simulation_timestamp.strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join(self._save_path, f'env_render_step_{step_count}_{timestamp_str}.png')
    try:
      image.save(img_path)
      logger.info('Saved environment render to %s', img_path)
    except Exception as e: # pylint: disable=broad-except
      logger.error("Failed to save environment render: %s", e)


  def __call__(self, trajectory: trajectory_lib.Trajectory) -> None:
    """Processes a trajectory and, if the interval is met, renders and plots.

    This method is called for each step the agent takes. It accumulates rewards
    and increments a counter. When the counter reaches `render_interval_steps`,
    it triggers environment rendering and metrics plotting.

    Args:
      trajectory: The `tf_agents.trajectories.trajectory.Trajectory` from the
        current step.
    """
    logger.debug('RenderingObserver called at step %d.', self._counter)

    try:
      reward_value = float(trajectory.reward)
    except TypeError:
      if hasattr(trajectory.reward, 'numpy') and trajectory.reward.numpy().size == 1:
        reward_value = float(trajectory.reward.numpy().item())
      else:
        logger.warning("Cannot extract scalar reward for RenderingObserver.")
        reward_value = 0.0
    self._cumulative_reward += reward_value
    self._counter += 1

    if self._start_time is None:
      self._start_time = pd.Timestamp.now()

    if self._counter % self._render_interval_steps == 0 and self._environment:
      logger.info('Rendering/plotting at step %d...', self._counter)
      current_pd_time = pd.Timestamp.now()
      execution_time_delta = current_pd_time - self._start_time
      mean_execution_time_sec = execution_time_delta.total_seconds() / self._counter

      logger.info(
          'Step %d: Cumulative reward = %.2f, Mean execution time = %.2fs/step',
          self._counter, float(self._cumulative_reward), mean_execution_time_sec
      )

      # Ensure the environment has the expected structure
      # This part is critical and relies on the specific env structure
      actual_env_for_render_plot = None
      if hasattr(self._environment, 'pyenv'):
          inner_env = self._environment.pyenv
          if isinstance(inner_env, list) and inner_env:
              actual_env_for_render_plot = inner_env[0]
          elif hasattr(inner_env, 'envs') and inner_env.envs: # TFPyEnvironment
              actual_env_for_render_plot = inner_env.envs[0]
          elif isinstance(inner_env, smart_control_env.Environment): # Direct smart_control Env
              actual_env_for_render_plot = inner_env


      if actual_env_for_render_plot and isinstance(actual_env_for_render_plot, smart_control_env.Environment):
        if actual_env_for_render_plot.metrics_path:
          logger.info('Plotting timeseries charts from metrics path: %s', actual_env_for_render_plot.metrics_path)
          try:
            reader = get_latest_episode_reader(actual_env_for_render_plot.metrics_path)
            if reader:
              self._plot_fn(reader, self._time_zone, self._counter) # Use self._plot_fn
            else:
              logger.warning("Failed to get metrics reader, skipping plotting.")
          except Exception as e: # pylint: disable=broad-except
            logger.error("Error during plotting: %s", e)
        else:
          logger.info("No metrics_path in environment, skipping timeseries plotting.")

        try:
          self._render_fn(actual_env_for_render_plot, self._counter) # Use self._render_fn
        except Exception as e: # pylint: disable=broad-except
          logger.error("Error during rendering: %s", e)
      else:
        logger.warning("Environment not configured correctly for rendering/plotting, or is None.")


  def reset(self) -> None:
    """Resets the observer's internal state for a new episode.

    This clears the step counter, cumulative reward, and resets the start time.
    """
    self._counter = 0
    self._cumulative_reward = 0.0
    self._start_time = None
