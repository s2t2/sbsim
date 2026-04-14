from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import numpy as np

from smart_control.utils import factory_utils
from smart_control.proto import smart_control_building_pb2
from smart_control.environment import environment_test_utils
from smart_control.simulator import building
from smart_control.utils import conversion_utils

class FactoryUtilsTest(parameterized.TestCase):

    def test_action_request_factory_variations(self):
        # Default instance
        default_ar = factory_utils.ActionRequestFactory()
        self.assertIsInstance(default_ar, smart_control_building_pb2.ActionRequest)
        self.assertEqual(len(default_ar.single_action_requests), 0)

        # With a specific timestamp
        custom_timestamp_pd = pd.Timestamp("2024-01-01 12:00:00")
        custom_timestamp_proto = conversion_utils.pandas_to_proto_timestamp(custom_timestamp_pd)
        ar_custom_ts = factory_utils.ActionRequestFactory(timestamp=custom_timestamp_proto)
        self.assertEqual(ar_custom_ts.timestamp, custom_timestamp_proto)

        # With multiple single action requests
        sar1 = factory_utils.SingleActionRequestFactory(
            device_id="boiler_complex",
            setpoint_name="temp_set",
            continuous_value=75.0
        )
        sar2 = factory_utils.SingleActionRequestFactory(
            device_id="vav_complex",
            setpoint_name="flow_rate",
            continuous_value=120.5
        )
        ar_with_sars = factory_utils.ActionRequestFactory(
            single_action_requests=[sar1, sar2]
        )
        self.assertEqual(len(ar_with_sars.single_action_requests), 2)
        self.assertEqual(ar_with_sars.single_action_requests[0].device_id, "boiler_complex")
        self.assertEqual(ar_with_sars.single_action_requests[1].continuous_value, 120.5)

    def test_simple_building_factory_variations(self):
        # Default instance
        default_sb = factory_utils.SimpleBuildingFactory()
        self.assertIsInstance(default_sb, environment_test_utils.SimpleBuilding)
        self.assertEqual(default_sb.start_time, pd.Timestamp("2022-03-13 00:00:00", tz="UTC"))
        self.assertIn("setpoint_1", default_sb.setpoint_names)

        # With custom start_time and setpoint_names
        custom_start_time = pd.Timestamp("2023-05-10 08:00:00", tz="America/New_York")
        custom_setpoints = ["custom_sp_1", "custom_sp_2"]
        sb_custom = factory_utils.SimpleBuildingFactory(
            start_time=custom_start_time,
            setpoint_names=custom_setpoints
        )
        self.assertEqual(sb_custom.start_time, custom_start_time)
        self.assertEqual(sb_custom.setpoint_names, custom_setpoints)
        # Check that values dict is updated based on new measurement_names (if logic is tied)
        # or ensure measurement_names can also be customized.
        # For SimpleBuilding, `values` are initialized with measurement_names.
        # Let's test with custom measurement_names too.
        custom_measurements = ["custom_m_1"]
        sb_custom_measure = factory_utils.SimpleBuildingFactory(
            measurement_names=custom_measurements
        )
        self.assertEqual(sb_custom_measure.measurement_names, custom_measurements)
        for m_name in custom_measurements:
            self.assertIn(m_name, sb_custom_measure.values)


    def test_floor_plan_building_factory_variations(self):
        # Default instance
        default_fpb = factory_utils.FloorPlanBasedBuildingFactory()
        self.assertIsInstance(default_fpb, building.FloorPlanBasedBuilding)
        self.assertEqual(default_fpb.cv_size_cm, 20.0)

        # With a custom floor plan
        custom_plan = np.array([
            [2, 2, 2, 2],
            [2, 0, 0, 2], # A small room
            [2, 0, 0, 2],
            [2, 2, 2, 2],
        ], dtype=np.int32)
        
        fpb_custom_plan = factory_utils.FloorPlanBasedBuildingFactory(
            floor_plan=custom_plan,
            zone_map=custom_plan # Assuming zone_map matches for simplicity here
        )
        np.testing.assert_array_equal(fpb_custom_plan.floor_plan, custom_plan)
        self.assertEqual(fpb_custom_plan.initial_temp, 292.0) # Default initial_temp

        # With custom initial temperature and properties
        custom_temp = 300.0
        custom_air_props = factory_utils.MaterialPropertiesFactory(conductivity=60.0)
        fpb_custom_props = factory_utils.FloorPlanBasedBuildingFactory(
            initial_temp=custom_temp,
            inside_air_properties=custom_air_props
        )
        self.assertEqual(fpb_custom_props.initial_temp, custom_temp)
        self.assertEqual(fpb_custom_props.inside_air_properties.conductivity, 60.0)


if __name__ == '__main__':
    absltest.main()
