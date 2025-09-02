import pytest
import numpy as np
from smart_control.simulator import air_handler
from smart_control.simulator import boiler
from smart_control.simulator import building
from smart_control.simulator import hvac
from smart_control.simulator import setpoint_schedule

@pytest.fixture
def default_air_properties() -> building.MaterialProperties:
  """Returns default material properties for air."""
  return building.MaterialProperties(
      specific_heat=1000.0,  # J/kgK
      density=1.2,  # kg/m3
      thermal_conductivity=0.025,  # W/mK
      thickness=1.0,  # m
  )

@pytest.fixture
def default_wall_properties() -> building.MaterialProperties:
  """Returns default material properties for internal walls."""
  return building.MaterialProperties(
      specific_heat=1000.0,  # J/kgK
      density=2000.0,  # kg/m3
      thermal_conductivity=1.0,  # W/mK
      thickness=0.1,  # m
  )

@pytest.fixture
def default_exterior_properties() -> building.MaterialProperties:
  """Returns default material properties for exterior surfaces."""
  return building.MaterialProperties(
      specific_heat=1000.0,  # J/kgK
      density=2500.0,  # kg/m3
      thermal_conductivity=1.5,  # W/mK
      thickness=0.2,  # m
  )

@pytest.fixture
def dummy_floor_plan() -> np.ndarray:
  """Returns a default numpy array for a floor plan."""
  return np.array([
      [1, 1, 1, 1, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 0, 0, 0, 1],
      [1, 1, 1, 1, 1],
  ])

@pytest.fixture
def dummy_zone_map() -> np.ndarray:
  """Returns a default numpy array for a zone map."""
  return np.array([
      [1, 1, 1, 1, 1],
      [1, 2, 2, 2, 1],
      [1, 2, 2, 2, 1],
      [1, 2, 2, 2, 1],
      [1, 1, 1, 1, 1],
  ])

@pytest.fixture
def default_floor_plan_building(
    dummy_floor_plan: np.ndarray,
    default_air_properties: building.MaterialProperties,
    default_wall_properties: building.MaterialProperties,
    default_exterior_properties: building.MaterialProperties,
) -> building.FloorPlanBasedBuilding:
  """Returns a FloorPlanBasedBuilding instance."""
  return building.FloorPlanBasedBuilding(
      floor_plan=dummy_floor_plan,
      zone_map=dummy_floor_plan,  # Use same for simplicity
      air_properties=default_air_properties,
      internal_wall_properties=default_wall_properties,
      external_wall_properties=default_exterior_properties,
      floor_properties=default_exterior_properties,
      ceiling_properties=default_exterior_properties,
      zone_height=2.5,  # m
      grid_size=1.0,  # m
  )

@pytest.fixture
def default_legacy_building() -> building.Building:
  """Returns a legacy Building instance."""
  return building.Building(
      number_zones=1,
      zone_areas_m2=[100.0],
      zone_heights_m=[3.0],
      external_wall_lengths_m=[40.0],
      external_window_lengths_m=[20.0],
      internal_wall_lengths_m=[0.0],
      thermal_conductivity_walls_w_per_mk=[0.1],
      thermal_conductivity_windows_w_per_mk=[0.8],
      wall_thickness_m=[0.2],
      initial_zone_temperatures_c=[20.0],
      initial_wall_temperatures_c=[20.0],
  )

@pytest.fixture
def default_air_handler_params() -> dict:
  """Returns default parameters for an AirHandler."""
  return {
      "recirculation_fraction": 0.5,
      "fan_efficiency": 0.6,
      "fan_pressure_rise_pa": 500.0,
      "heating_air_temperature_setpoint_c": 40.0,
      "cooling_air_temperature_setpoint_c": 10.0,
      "max_heating_power_w": 100000.0,
      "max_cooling_power_w": 100000.0,
      "cooling_coil_cop": 3.0,
      "economizer_max_fraction": 1.0,
  }

@pytest.fixture
def default_air_handler(
    default_air_handler_params: dict,
) -> air_handler.AirHandler:
  """Returns an AirHandler instance with default parameters."""
  return air_handler.AirHandler(**default_air_handler_params)

@pytest.fixture
def default_boiler() -> boiler.Boiler:
  """Returns a Boiler instance with default parameters."""
  return boiler.Boiler(
      max_power_w=100000.0,
      efficiency=0.9,
      water_temperature_setpoint_c=60.0,
  )

@pytest.fixture
def default_setpoint_schedule() -> setpoint_schedule.SetpointSchedule:
  """Returns a SetpointSchedule instance with default parameters."""
  return setpoint_schedule.SetpointSchedule(
      occupied_heating_setpoint_c=20.0,
      occupied_cooling_setpoint_c=25.0,
      unoccupied_heating_setpoint_c=18.0,
      unoccupied_cooling_setpoint_c=28.0,
  )

@pytest.fixture
def default_hvac(
    default_air_handler: air_handler.AirHandler,
    default_boiler: boiler.Boiler,
    default_setpoint_schedule: setpoint_schedule.SetpointSchedule,
) -> hvac.Hvac:
  """Returns an Hvac instance with default components."""
  return hvac.Hvac(
      air_handler=default_air_handler,
      boiler=default_boiler,
      setpoint_schedule=default_setpoint_schedule,
      zone_coordinates_from_zone_index={0: (0, 0)},
  )

@pytest.fixture
def common_air_handler_test_params() -> dict:
  """Returns common parameters for AirHandler tests."""
  return {
      "recirculation_fraction": 0.6,
      "heating_air_temperature_setpoint_c": 35.0,
      "cooling_air_temperature_setpoint_c": 15.0,
      "fan_efficiency": 0.7,
      "fan_pressure_rise_pa": 600.0,
      "max_heating_power_w": 50000.0,
      "max_cooling_power_w": 50000.0,
      "cooling_coil_cop": 3.5,
      "economizer_max_fraction": 0.8,
  }
