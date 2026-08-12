import factory
import pandas as pd # Adjusted pandas import
import numpy as np # Added import
from smart_control.proto import smart_control_building_pb2
from smart_control.utils import conversion_utils # For pandas_to_proto_timestamp
from smart_control.environment import environment_test_utils
from smart_control.simulator import building # Added for building.MaterialProperties etc.

class ActionRequestFactory(factory.Factory):
    class Meta:
        model = smart_control_building_pb2.ActionRequest

    timestamp = factory.LazyFunction(lambda: conversion_utils.pandas_to_proto_timestamp(pd.Timestamp("2023-10-26 10:00:00")))
    
    # single_action_requests field allows passing a list of SingleActionRequest objects when creating an ActionRequest
    single_action_requests = factory.List([])

    @factory.post_generation
    def fill_single_action_requests(obj, create, results=None):
        # This hook processes the 'single_action_requests' list passed during factory creation.
        # 'results' here will be the list provided to 'single_action_requests' parameter.
        if results:
            obj.single_action_requests.extend(results)

class SingleActionRequestFactory(factory.Factory):
    class Meta:
        model = smart_control_building_pb2.SingleActionRequest

    device_id = factory.Sequence(lambda n: f"device_{n}")
    setpoint_name = factory.Sequence(lambda n: f"setpoint_{n}")
    continuous_value = factory.Faker('pyfloat', left_digits=2, right_digits=2, positive=True, min_value=10.0, max_value=30.0)
    # discrete_value, string_value, boolean_value can be added if needed

# Example of how to use it with ActionRequestFactory:
# action_request = ActionRequestFactory(
#     add_single_request=[
#         SingleActionRequestFactory(device_id="boiler_1", setpoint_name="temperature", continuous_value=25.5),
#         SingleActionRequestFactory(device_id="vav_1", setpoint_name="flow", continuous_value=100.0)
#     ]
# )
# Or by building them separately:
# sar1 = SingleActionRequestFactory()
# action_request = ActionRequestFactory(single_action_requests=[sar1])


class SingleObservationRequestFactory(factory.Factory):
    class Meta:
        model = smart_control_building_pb2.SingleObservationRequest

    device_id = factory.Sequence(lambda n: f"obs_device_{n}")
    measurement_name = factory.Sequence(lambda n: f"obs_measurement_{n}")

class ObservationRequestFactory(factory.Factory):
    class Meta:
        model = smart_control_building_pb2.ObservationRequest
    
    timestamp = factory.LazyFunction(lambda: conversion_utils.pandas_to_proto_timestamp(pd.Timestamp("2023-10-26 10:00:00")))
    # single_observation_requests can be built using a post_generation hook or by passing a list of SingleObservationRequest instances

class SingleObservationResponseFactory(factory.Factory):
    class Meta:
        model = smart_control_building_pb2.SingleObservationResponse

    single_observation_request = factory.SubFactory(SingleObservationRequestFactory)
    timestamp = factory.LazyFunction(lambda: conversion_utils.pandas_to_proto_timestamp(pd.Timestamp("2023-10-26 10:00:00")))
    observation_valid = True
    continuous_value = factory.Faker('pyfloat', left_digits=2, right_digits=2, positive=True, min_value=60.0, max_value=80.0)
    # discrete_value, string_value, boolean_value can be added if needed

class ObservationResponseFactory(factory.Factory):
    class Meta:
        model = smart_control_building_pb2.ObservationResponse

    timestamp = factory.LazyFunction(lambda: conversion_utils.pandas_to_proto_timestamp(pd.Timestamp("2023-10-26 10:00:00")))
    request = factory.SubFactory(ObservationRequestFactory) # Link to ObservationRequestFactory
    # single_observation_responses can be built using a post_generation hook or by passing a list of SingleObservationResponse instances
    # Example:
    # @factory.post_generation
    # def add_single_observation_responses(obj, create, results=None):
    #     if not create:
    #         return
    #     if results: # results is a list of SingleObservationResponse instances
    #         for res in results:
    #             obj.single_observation_responses.append(res)


class SimpleBuildingFactory(factory.Factory):
    class Meta:
        model = environment_test_utils.SimpleBuilding

    # SimpleBuilding constructor takes:
    # start_time: pd.Timestamp = pd.Timestamp("2022-03-13 00:00:00", tz="UTC"),
    # step_duration: pd.Timedelta = pd.Timedelta(minutes=10),
    # setpoint_names: Optional[list[str]] = None,
    # measurement_names: Optional[list[str]] = None,

    start_time = pd.Timestamp("2022-03-13 00:00:00", tz="UTC")
    step_duration = pd.Timedelta(minutes=10)
    setpoint_names = factory.List([
        "setpoint_1", "setpoint_2", "setpoint_3", "setpoint_4", "setpoint_5", "setpoint_6"
    ])
    measurement_names = factory.List([
        "measurement_1", "measurement_2", "measurement_3", "measurement_4", "measurement_5"
    ])

    # This factory will initialize SimpleBuilding with some default values.
    # The `values` dictionary in SimpleBuilding is initialized within its __init__
    # based on measurement_names.


# Factory for MaterialProperties
class MaterialPropertiesFactory(factory.Factory):
    class Meta:
        model = building.MaterialProperties

    conductivity = factory.Faker('pyfloat', left_digits=1, right_digits=2, positive=True, min_value=0.01, max_value=50.0)
    heat_capacity = factory.Faker('pyfloat', left_digits=3, right_digits=1, positive=True, min_value=700.0, max_value=1200.0)
    density = factory.Faker('pyfloat', left_digits=3, right_digits=1, positive=True, min_value=1.0, max_value=3000.0)

# Factory for the deprecated building.Building
class DeprecatedBuildingFactory(factory.Factory):
    class Meta:
        model = building.Building

    cv_size_cm = 20.0
    floor_height_cm = 300.0
    room_shape = (3, 2) # Default room_shape (rows, cols) for CVs
    building_shape = (2, 3) # Default building_shape (rooms_x, rooms_y)
    initial_temp = 292.0
    inside_air_properties = factory.SubFactory(MaterialPropertiesFactory, conductivity=50.0, heat_capacity=700.0, density=1.0)
    inside_wall_properties = factory.SubFactory(MaterialPropertiesFactory, conductivity=2.0, heat_capacity=1000.0, density=1800.0)
    building_exterior_properties = factory.SubFactory(MaterialPropertiesFactory, conductivity=0.05, heat_capacity=1000.0, density=3000.0)

# Factory for building.FloorPlanBasedBuilding
class FloorPlanBasedBuildingFactory(factory.Factory):
    class Meta:
        model = building.FloorPlanBasedBuilding

    cv_size_cm = 20.0
    floor_height_cm = 300.0
    initial_temp = 292.0
    inside_air_properties = factory.SubFactory(MaterialPropertiesFactory, conductivity=50.0, heat_capacity=700.0, density=1.0)
    inside_wall_properties = factory.SubFactory(MaterialPropertiesFactory, conductivity=2.0, heat_capacity=1000.0, density=1800.0)
    building_exterior_properties = factory.SubFactory(MaterialPropertiesFactory, conductivity=0.05, heat_capacity=1000.0, density=3000.0)
    
    floor_plan_filepath = None # Must be None if floor_plan array is provided
    zone_map_filepath = None   # Must be None if zone_map array is provided
    buffer_from_walls = 0
    diffuser_spacing = 10 # A default value, can be overridden

    # Default floor_plan and zone_map (simple 1-room)
    # Users can override these with more complex np.arrays
    floor_plan = factory.LazyFunction(lambda: np.array([
        [2, 2, 2, 2, 2],
        [2, 1, 0, 1, 2], # 0 is room, 1 is interior wall, 2 is exterior wall
        [2, 1, 0, 1, 2],
        [2, 1, 0, 1, 2],
        [2, 2, 2, 2, 2],
    ], dtype=np.int32))
    
    zone_map = factory.LazyAttribute(lambda o: o.floor_plan) # By default, zone_map is same as floor_plan

    # For stochastic convection simulator, if needed by tests
    # convection_simulator = None
