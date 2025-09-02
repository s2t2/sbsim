import pytest
import pandas as pd

from smart_control.reward import base_setpoint_energy_carbon_reward
from smart_control.utils import conversion_utils
from smart_control.utils import rewards_utils
from smart_control.event import event_pb2 as smart_control_event_pb2
from smart_control.common import time_utils

# TODO(b/265342085): remove the following import and update proto path.
#from google3.experimental.users.smarts.control.proto import reward_pb2 as smart_control_reward_pb2
from smart_control.reward import reward_pb2 as smart_control_reward_pb2


@pytest.fixture
def base_setpoint_reward_func() -> base_setpoint_energy_carbon_reward.BaseSetpointEnergyCarbonRewardFunction:
  """Returns a BaseSetpointEnergyCarbonRewardFunction instance."""
  return base_setpoint_energy_carbon_reward.BaseSetpointEnergyCarbonRewardFunction(
      comfort_temperature_low_c=20.0,
      comfort_temperature_high_c=25.0,
      comfort_penalty_multiplier=1.0,
      energy_cost_multiplier=0.0001,
      carbon_emission_multiplier=0.001,
      action_smoothing_multiplier=0.01,
      energy_cost_name_to_calculator_map={}, # Using empty for base tests
      carbon_emission_name_to_calculator_map={}, # Using empty for base tests
  )

@pytest.fixture
def sample_reward_info() -> smart_control_reward_pb2.RewardInfo:
  """Returns a populated RewardInfo object for testing."""
  start_time = pd.Timestamp("2023-01-01 10:00:00")
  end_time = pd.Timestamp("2023-01-01 11:00:00")
  zone_ids = ["zone_1", "zone_2"]
  reward_info = smart_control_reward_pb2.RewardInfo()

  # Populate AgentEpisodeStats
  reward_info.agent_episode_stats.episode_start_timestamp.CopyFrom(
      time_utils.seconds_from_pd_timestamp(start_time)
  )
  reward_info.agent_episode_stats.episode_end_timestamp.CopyFrom(
      time_utils.seconds_from_pd_timestamp(end_time)
  )
  reward_info.agent_episode_stats.total_reward = 100.0
  reward_info.agent_episode_stats.total_comfort_penalty = 10.0
  reward_info.agent_episode_stats.total_energy_cost = 5.0
  reward_info.agent_episode_stats.total_carbon_emission_cost = 2.0
  reward_info.agent_episode_stats.total_action_smoothing_cost = 1.0

  # Populate ZoneDatapoint for each zone
  for zone_id in zone_ids:
    zone_datapoint = reward_info.zone_datapoints.add()
    zone_datapoint.zone_id = zone_id
    zone_datapoint.comfort_penalty = 5.0 # Example value
    zone_datapoint.temperature_c = 22.0 # Example value
    # Example setpoint, assuming it's a simple value for this test object
    zone_datapoint.setpoint_temperature_c = 23.0

  # Populate ControlDatapoint (This part might not be what the original tests used directly for sum operations)
  # The original tests for sum_electricity_energy_rate etc. likely relied on specific fields within
  # zone_reward_infos, air_handler_reward_infos, and boiler_reward_infos.
  # Let's add those specific structures as per the old _get_test_reward_info.

  # ZoneRewardInfo (mimicking the structure from the old test file)
  # Assuming two zones '0,0' and '1,1' as in the old test
  zone_ids_for_specific_fields = ['0,0', '1,1']
  for zone_id_str in zone_ids_for_specific_fields:
    zone_info = reward_info.zone_reward_infos[zone_id_str] # Access existing or create new
    zone_info.heating_setpoint_temperature = 293.0
    zone_info.cooling_setpoint_temperature = 297.0
    zone_info.zone_air_temperature = 294.0  # Example: in setpoint for productivity
    zone_info.average_occupancy = 5.0 # Example value from old test
    zone_info.air_flow_rate_setpoint = 0.013
    zone_info.air_flow_rate = 0.012

  # AirHandlerRewardInfo (mimicking the structure)
  ah_info = reward_info.air_handler_reward_infos['air_handler_0'] # Access existing or create new
  ah_info.blower_electrical_energy_rate = 800.0
  ah_info.air_conditioning_electrical_energy_rate = 4500.0

  # BoilerRewardInfo (mimicking the structure)
  b_info = reward_info.boiler_reward_infos['boiler_0'] # Access existing or create new
  b_info.natural_gas_heating_energy_rate = 5000.0
  b_info.pump_electrical_energy_rate = 250.0
  
  # The ControlDatapoint added by the previous version of this fixture might be useful for other tests,
  # but the specific sum tests relied on the above. We can keep it or remove it if it causes issues.
  # For now, let's keep it but ensure the above structures are primary for the sum tests.

  # Populate ControlDatapoint (as originally in this fixture, might be for different tests)
  control_datapoint = reward_info.control_datapoints.add()
  control_datapoint.timestamp.CopyFrom(
      time_utils.seconds_from_pd_timestamp(end_time) # Example timestamp
  )
  control_datapoint.comfort_penalty = 10.0 # This is an aggregated value
  control_datapoint.energy_cost = 5.0 # Aggregated
  control_datapoint.carbon_emission_cost = 2.0 # Aggregated
  control_datapoint.action_smoothing_cost = 1.0 # Aggregated
  control_datapoint.total_reward = 82.0 # Aggregated

  # Populate EnergyCostDatapoint for the control_datapoint
  energy_cost_datapoint = control_datapoint.energy_cost_breakdown.add()
  energy_cost_datapoint.name = "electricity"
  energy_cost_datapoint.cost = 3.0
  energy_cost_datapoint.energy_gj = 0.1 # This is total energy, not rate.
  energy_cost_datapoint = control_datapoint.energy_cost_breakdown.add()
  energy_cost_datapoint.name = "natural_gas"
  energy_cost_datapoint.cost = 2.0
  energy_cost_datapoint.energy_gj = 0.05 # This is total energy, not rate.

  # Populate CarbonEmissionDatapoint
  carbon_emission_datapoint = control_datapoint.carbon_emission_breakdown.add()
  carbon_emission_datapoint.name = "electricity"
  carbon_emission_datapoint.carbon_kg_co2e = 1.5
  carbon_emission_datapoint = control_datapoint.carbon_emission_breakdown.add()
  carbon_emission_datapoint.name = "natural_gas"
  carbon_emission_datapoint.carbon_kg_co2e = 0.5

  # Populate ActionSmoothingDatapoint
  action_smoothing_datapoint = control_datapoint.action_smoothing_breakdown.add()
  action_smoothing_datapoint.name = "thermostat_setpoint_change"
  action_smoothing_datapoint.cost = 1.0
  
  # Set start and end time for the RewardInfo itself, not just AgentEpisodeStats
  # This is what _get_delta_time_sec uses.
  reward_info.start_timestamp.CopyFrom(time_utils.seconds_from_pd_timestamp(pd.Timestamp('2021-05-03 12:13:00-05:00', tz='US/Eastern')))
  reward_info.end_timestamp.CopyFrom(time_utils.seconds_from_pd_timestamp(pd.Timestamp('2021-05-03 12:18:00-05:00', tz='US/Eastern')))


  return reward_info

@pytest.fixture
def default_electricity_cost_calculator() -> electricity_energy_cost.ElectricityEnergyCost:
  """Returns an ElectricityEnergyCost instance with default parameters."""
  return electricity_energy_cost.ElectricityEnergyCost(
      price_per_kwh=0.15,  # Example price
      carbon_intensity_kg_co2e_per_kwh=0.5,  # Example intensity
  )

@pytest.fixture
def default_natural_gas_cost_calculator() -> natural_gas_energy_cost.NaturalGasEnergyCost:
  """Returns a NaturalGasEnergyCost instance with default parameters."""
  # The NaturalGasEnergyCost constructor expects gas_price_by_month.
  # Using the default GAS_PRICE_BY_MONTH_SOURCE from the class itself.
  return natural_gas_energy_cost.NaturalGasEnergyCost()
