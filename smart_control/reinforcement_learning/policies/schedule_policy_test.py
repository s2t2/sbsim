import unittest
from unittest.mock import MagicMock # Not strictly needed yet, but good to have

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from smart_control.reinforcement_learning.policies import schedule_policy
from tf_agents.specs import array_spec # For action_spec when setpoint_range is used
from tf_agents.specs import tensor_spec # For general specs
from tf_agents.trajectories import time_step as ts_lib # Renamed to avoid conflict
from tf_agents.policies import policy_step


class SchedulePolicyTest(tf.test.TestCase):

    def _create_specs(self, num_actions=1, is_ventilation=False):
        obs_spec_dict = {
            'hour_of_day': tf.TensorSpec(shape=(), dtype=tf.float32, name='hour_of_day'),
            'day_of_week': tf.TensorSpec(shape=(), dtype=tf.float32, name='day_of_week'),
            'outdoor_temp': tf.TensorSpec(shape=(), dtype=tf.float32, name='outdoor_temp'), # Example other feature
        }
        if is_ventilation:
            # Ventilation might have specific observation needs, adjust if necessary
            pass
        
        time_step_spec = ts_lib.time_step_spec(obs_spec_dict)
        
        # Action spec for heating/cooling is typically a single float (setpoint)
        # Action spec for ventilation might be multi-dimensional
        action_shape = (num_actions,)
        action_spec = tf.TensorSpec(shape=action_shape, dtype=tf.float32, name='action')
        
        return time_step_spec, action_spec

    def setUp(self):
        super().setUp()
        self.heating_schedule = {
            # Day 0 (Sunday)
            0: {0: 18.0, 1: 18.0, 6: 20.0, 8: 21.0, 18: 21.0, 22: 19.0, 23: 18.0},
            # Day 1 (Monday)
            1: {0: 18.0, 1: 18.0, 6: 20.5, 8: 21.5, 18: 21.5, 22: 19.5, 23: 18.0},
            # Day 6 (Saturday) - Default for missing days
            6: {0: 19.0, 7: 22.0, 23: 19.0}
        }
        self.cooling_schedule = {
            0: {0: 28.0, 1: 28.0, 6: 26.0, 8: 25.0, 18: 25.0, 22: 27.0, 23: 28.0},
            1: {0: 28.0, 1: 28.0, 6: 26.5, 8: 25.5, 18: 25.5, 22: 27.5, 23: 28.0},
        }
        # Ventilation schedule has 2 actions: [outdoor_air_ratio, supply_fan_speed]
        self.ventilation_schedule = {
            0: {0: [0.1, 80.0], 7: [0.3, 100.0], 17: [0.3, 100.0], 19: [0.2, 90.0]},
            1: {0: [0.1, 80.0], 7: [0.4, 120.0], 17: [0.4, 120.0], 19: [0.2, 90.0]},
        }
        self.default_setpoint_range = (15.0, 30.0) # For heating/cooling
        self.default_ventilation_action_spec_num_actions = 2

    def test_init_heating_schedule(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='heating',
            heating_setpoint_schedule=self.heating_schedule
        )
        self.assertEqual(policy._schedule_type, 'heating')
        self.assertEqual(policy._action_spec, action_spec)
        self.assertEqual(policy._time_step_spec, time_step_spec)
        self.assertIs(policy._schedule, self.heating_schedule)
        self.assertEqual(policy._num_actions, 1)

    def test_init_cooling_schedule(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='cooling',
            cooling_setpoint_schedule=self.cooling_schedule
        )
        self.assertEqual(policy._schedule_type, 'cooling')
        self.assertIs(policy._schedule, self.cooling_schedule)

    def test_init_ventilation_schedule(self):
        time_step_spec, action_spec = self._create_specs(num_actions=self.default_ventilation_action_spec_num_actions)
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='ventilation',
            ventilation_action_schedule=self.ventilation_schedule
        )
        self.assertEqual(policy._schedule_type, 'ventilation')
        self.assertIs(policy._schedule, self.ventilation_schedule)
        self.assertEqual(policy._num_actions, self.default_ventilation_action_spec_num_actions)

    def test_init_action_spec_from_setpoint_range(self):
        time_step_spec, _ = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=None, # Provide None for action_spec
            schedule_type='heating',
            heating_setpoint_schedule=self.heating_schedule,
            setpoint_range=self.default_setpoint_range
        )
        self.assertIsInstance(policy._action_spec, array_spec.BoundedArraySpec)
        self.assertEqual(policy._action_spec.shape, (1,))
        self.assertEqual(policy._action_spec.minimum, self.default_setpoint_range[0])
        self.assertEqual(policy._action_spec.maximum, self.default_setpoint_range[1])
        self.assertEqual(policy._action_spec.dtype, np.float32)

    def test_init_action_spec_from_setpoint_range_no_schedule_type(self):
        time_step_spec, _ = self._create_specs()
        # This case is valid if schedule_type is None and setpoint_range is given
        # The policy might default to a generic schedule or simply use the range for clipping
        # For SchedulePolicy, it expects a schedule, so this might be an edge case or invalid setup
        # Depending on the class's design, it might raise an error or create a default schedule.
        # Given the current structure, it will likely fail when trying to get a schedule.
        # Let's assume it should raise an error if schedule_type is None but a schedule is expected.
        # However, the problem states "action_spec derived from setpoint_range", not a full policy test.
        action_spec = schedule_policy.SchedulePolicy._get_action_spec_from_setpoint_range(
            setpoint_range=self.default_setpoint_range,
            num_actions=1 # Assuming single action for setpoint range
        )
        self.assertIsInstance(action_spec, array_spec.BoundedArraySpec)
        self.assertEqual(action_spec.shape, (1,))
        self.assertEqual(action_spec.minimum, self.default_setpoint_range[0])
        self.assertEqual(action_spec.maximum, self.default_setpoint_range[1])


    def test_init_invalid_schedule_type(self):
        time_step_spec, action_spec = self._create_specs()
        with self.assertRaisesRegex(ValueError, "schedule_type must be one of"):
            schedule_policy.SchedulePolicy(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                schedule_type='unknown_type',
                heating_setpoint_schedule=self.heating_schedule
            )

    def test_init_missing_heating_schedule(self):
        time_step_spec, action_spec = self._create_specs()
        with self.assertRaisesRegex(ValueError, "heating_setpoint_schedule must be provided"):
            schedule_policy.SchedulePolicy(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                schedule_type='heating',
                heating_setpoint_schedule=None # Missing
            )

    def test_init_missing_cooling_schedule(self):
        time_step_spec, action_spec = self._create_specs()
        with self.assertRaisesRegex(ValueError, "cooling_setpoint_schedule must be provided"):
            schedule_policy.SchedulePolicy(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                schedule_type='cooling',
                cooling_setpoint_schedule=None # Missing
            )

    def test_init_missing_ventilation_schedule(self):
        time_step_spec, action_spec = self._create_specs(num_actions=self.default_ventilation_action_spec_num_actions)
        with self.assertRaisesRegex(ValueError, "ventilation_action_schedule must be provided"):
            schedule_policy.SchedulePolicy(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                schedule_type='ventilation',
                ventilation_action_schedule=None # Missing
            )

    def test_init_no_action_spec_and_no_setpoint_range(self):
        time_step_spec, _ = self._create_specs()
        with self.assertRaisesRegex(ValueError, "Either action_spec or setpoint_range must be provided."):
            schedule_policy.SchedulePolicy(
                time_step_spec=time_step_spec,
                action_spec=None,
                setpoint_range=None, # Both None
                schedule_type='heating',
                heating_setpoint_schedule=self.heating_schedule
            )

    def _create_time_step(self, hour, day, outdoor_temp=20.0, is_batched=False, batch_size=1):
        observation = {
            'hour_of_day': tf.constant(hour, dtype=tf.float32),
            'day_of_week': tf.constant(day, dtype=tf.float32),
            'outdoor_temp': tf.constant(outdoor_temp, dtype=tf.float32),
        }
        if is_batched:
            for k, v in observation.items():
                observation[k] = tf.stack([v] * batch_size)
        # For single timestep, TimeStep expects observation to have a batch dim already
        # unless it's tf_agents.utils.nest_utils.batch_nested_tensors which is more complex
        # Simplest is to always have a batch dim of 1 for single and then stack for batched.
        if not is_batched: # Add batch dimension for single TimeStep
             for k, v in observation.items():
                observation[k] = tf.expand_dims(v, axis=0)
        
        return ts_lib.transition(observation, reward=tf.zeros(batch_size if is_batched else 1, dtype=tf.float32))


    def test_action_heating_single_timestep(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='heating',
            heating_setpoint_schedule=self.heating_schedule
        )

        # Case 1: Exact match in schedule (Monday 8:00 -> 21.5)
        time_step = self._create_time_step(hour=8.0, day=1.0)
        policy_step_output = policy.action(time_step)
        self.assertIsInstance(policy_step_output, policy_step.PolicyStep)
        self.assertAllClose(policy_step_output.action, tf.constant([[21.5]], dtype=tf.float32))

        # Case 2: Hour between scheduled hours (Monday 7:00 -> uses 6:00 value of 20.5)
        time_step = self._create_time_step(hour=7.0, day=1.0)
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[20.5]], dtype=tf.float32))
        
        # Case 3: Day not in schedule (Tuesday Day 2, uses Saturday Day 6 schedule, 7:00 -> 22.0)
        time_step = self._create_time_step(hour=7.0, day=2.0) # Day 2 (Tuesday)
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[22.0]], dtype=tf.float32))

        # Case 4: Hour not in schedule for a day (Sunday 5:00 -> uses Sunday 1:00 value of 18.0)
        time_step = self._create_time_step(hour=5.0, day=0.0) # Sunday 5:00
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[18.0]], dtype=tf.float32))

        # Case 5: Hour after last scheduled hour (Saturday 23:30 -> uses Saturday 23:00 value of 19.0)
        time_step = self._create_time_step(hour=23.5, day=6.0)
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[19.0]], dtype=tf.float32))

        # Case 6: Hour before first scheduled hour (Saturday 6:00 -> uses Saturday 0:00 value of 19.0)
        time_step = self._create_time_step(hour=6.0, day=6.0)
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[19.0]], dtype=tf.float32))

    def test_action_heating_batched_timestep(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='heating',
            heating_setpoint_schedule=self.heating_schedule
        )

        # Batch of 2:
        # 1. Monday 8:00 -> 21.5
        # 2. Sunday 7:00 (uses 6:00 value) -> 20.0
        time_step_batched = self._create_time_step(
            hour=tf.constant([8.0, 7.0], dtype=tf.float32),
            day=tf.constant([1.0, 0.0], dtype=tf.float32),
            is_batched=True, batch_size=2
        )
        policy_step_output = policy.action(time_step_batched)
        self.assertIsInstance(policy_step_output, policy_step.PolicyStep)
        self.assertEqual(policy_step_output.action.shape, (2,1))
        self.assertAllClose(policy_step_output.action, tf.constant([[21.5], [20.0]], dtype=tf.float32))

    def test_action_cooling_single_timestep(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='cooling',
            cooling_setpoint_schedule=self.cooling_schedule
        )
        # Monday 8:00 -> 25.5
        time_step = self._create_time_step(hour=8.0, day=1.0)
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[25.5]], dtype=tf.float32))

        # Day not in cooling schedule (e.g., Day 2 - Tuesday), should use default (Day 6 - Saturday)
        # Saturday doesn't exist in self.cooling_schedule, so it should use Day 0 from cooling_schedule
        # Sunday 7:00 (uses 6:00 value) -> 26.0
        time_step = self._create_time_step(hour=7.0, day=2.0) # Tuesday
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[26.0]], dtype=tf.float32))


    def test_action_cooling_batched_timestep(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='cooling',
            cooling_setpoint_schedule=self.cooling_schedule
        )
        # Batch of 2:
        # 1. Monday 8:00 -> 25.5
        # 2. Sunday 22:30 (uses 22:00 value) -> 27.0
        time_step_batched = self._create_time_step(
            hour=tf.constant([8.0, 22.5], dtype=tf.float32),
            day=tf.constant([1.0, 0.0], dtype=tf.float32),
            is_batched=True, batch_size=2
        )
        policy_step_output = policy.action(time_step_batched)
        self.assertEqual(policy_step_output.action.shape, (2,1))
        self.assertAllClose(policy_step_output.action, tf.constant([[25.5], [27.0]], dtype=tf.float32))

    def test_action_ventilation_single_timestep(self):
        time_step_spec, action_spec = self._create_specs(num_actions=self.default_ventilation_action_spec_num_actions)
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='ventilation',
            ventilation_action_schedule=self.ventilation_schedule
        )
        # Monday 7:00 -> [0.4, 120.0]
        time_step = self._create_time_step(hour=7.0, day=1.0)
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[0.4, 120.0]], dtype=tf.float32))

        # Day not in ventilation schedule (e.g., Day 2 - Tuesday), should use default (Day 6 - Saturday)
        # Day 6 doesn't exist in self.ventilation_schedule, so it should use Day 0.
        # Sunday 8:00 (uses 7:00 value) -> [0.3, 100.0]
        time_step = self._create_time_step(hour=8.0, day=2.0) # Tuesday
        policy_step_output = policy.action(time_step)
        self.assertAllClose(policy_step_output.action, tf.constant([[0.3, 100.0]], dtype=tf.float32))

    def test_action_ventilation_batched_timestep(self):
        time_step_spec, action_spec = self._create_specs(num_actions=self.default_ventilation_action_spec_num_actions)
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='ventilation',
            ventilation_action_schedule=self.ventilation_schedule
        )
        # Batch of 2:
        # 1. Monday 7:00 -> [0.4, 120.0]
        # 2. Sunday 19:30 (uses 19:00 value) -> [0.2, 90.0]
        time_step_batched = self._create_time_step(
            hour=tf.constant([7.0, 19.5], dtype=tf.float32),
            day=tf.constant([1.0, 0.0], dtype=tf.float32),
            is_batched=True, batch_size=2
        )
        policy_step_output = policy.action(time_step_batched)
        self.assertEqual(policy_step_output.action.shape, (2, self.default_ventilation_action_spec_num_actions))
        self.assertAllClose(policy_step_output.action, tf.constant([[0.4, 120.0], [0.2, 90.0]], dtype=tf.float32))

    def test_action_with_setpoint_range_clipping(self):
        time_step_spec, _ = self._create_specs() # action_spec will be derived
        # Schedule that goes outside the setpoint_range
        custom_heating_schedule = {0: {0: 10.0, 8: 35.0}} # Range is (15,30)
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=None,
            schedule_type='heating',
            heating_setpoint_schedule=custom_heating_schedule,
            setpoint_range=self.default_setpoint_range # (15.0, 30.0)
        )
        # Time: Sunday 0:00, schedule value 10.0, should be clipped to 15.0
        time_step1 = self._create_time_step(hour=0.0, day=0.0)
        policy_step1 = policy.action(time_step1)
        self.assertAllClose(policy_step1.action, tf.constant([[15.0]], dtype=tf.float32))

        # Time: Sunday 8:00, schedule value 35.0, should be clipped to 30.0
        time_step2 = self._create_time_step(hour=8.0, day=0.0)
        policy_step2 = policy.action(time_step2)
        self.assertAllClose(policy_step2.action, tf.constant([[30.0]], dtype=tf.float32))

    def test_distribution_heating_single_timestep(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='heating',
            heating_setpoint_schedule=self.heating_schedule
        )

        # Monday 8:00 -> action should be 21.5
        time_step = self._create_time_step(hour=8.0, day=1.0)
        distribution_step = policy.distribution(time_step)

        self.assertIsInstance(distribution_step, policy_step.PolicyStep)
        self.assertIsInstance(distribution_step.action, tfp.distributions.Deterministic)
        self.assertAllClose(distribution_step.action.loc, tf.constant([[21.5]], dtype=tf.float32))

    def test_distribution_heating_batched_timestep(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='heating',
            heating_setpoint_schedule=self.heating_schedule
        )

        # Batch of 2:
        # 1. Monday 8:00 -> 21.5
        # 2. Sunday 7:00 -> 20.0
        time_step_batched = self._create_time_step(
            hour=tf.constant([8.0, 7.0], dtype=tf.float32),
            day=tf.constant([1.0, 0.0], dtype=tf.float32),
            is_batched=True, batch_size=2
        )
        distribution_step = policy.distribution(time_step_batched)
        self.assertIsInstance(distribution_step.action, tfp.distributions.Deterministic)
        self.assertEqual(distribution_step.action.batch_shape, tf.TensorShape([2]))
        self.assertEqual(distribution_step.action.event_shape, tf.TensorShape([1]))
        self.assertAllClose(distribution_step.action.loc, tf.constant([[21.5], [20.0]], dtype=tf.float32))

    def test_distribution_cooling_single_timestep(self):
        time_step_spec, action_spec = self._create_specs()
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='cooling',
            cooling_setpoint_schedule=self.cooling_schedule
        )
        # Monday 8:00 -> 25.5
        time_step = self._create_time_step(hour=8.0, day=1.0)
        distribution_step = policy.distribution(time_step)
        self.assertIsInstance(distribution_step.action, tfp.distributions.Deterministic)
        self.assertAllClose(distribution_step.action.loc, tf.constant([[25.5]], dtype=tf.float32))

    def test_distribution_ventilation_single_timestep(self):
        time_step_spec, action_spec = self._create_specs(num_actions=self.default_ventilation_action_spec_num_actions)
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='ventilation',
            ventilation_action_schedule=self.ventilation_schedule
        )
        # Monday 7:00 -> [0.4, 120.0]
        time_step = self._create_time_step(hour=7.0, day=1.0)
        distribution_step = policy.distribution(time_step)
        self.assertIsInstance(distribution_step.action, tfp.distributions.Deterministic) # Should be Deterministic for schedule
        # For multi-dimensional action, loc will have that shape
        self.assertAllClose(distribution_step.action.loc, tf.constant([[0.4, 120.0]], dtype=tf.float32))

    def test_distribution_ventilation_batched_timestep(self):
        time_step_spec, action_spec = self._create_specs(num_actions=self.default_ventilation_action_spec_num_actions)
        policy = schedule_policy.SchedulePolicy(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            schedule_type='ventilation',
            ventilation_action_schedule=self.ventilation_schedule
        )
        # Batch of 2:
        # 1. Monday 7:00 -> [0.4, 120.0]
        # 2. Sunday 19:30 (uses 19:00 value) -> [0.2, 90.0]
        time_step_batched = self._create_time_step(
            hour=tf.constant([7.0, 19.5], dtype=tf.float32),
            day=tf.constant([1.0, 0.0], dtype=tf.float32),
            is_batched=True, batch_size=2
        )
        distribution_step = policy.distribution(time_step_batched)
        self.assertIsInstance(distribution_step.action, tfp.distributions.Deterministic)
        self.assertEqual(distribution_step.action.batch_shape, tf.TensorShape([2]))
        self.assertEqual(distribution_step.action.event_shape, tf.TensorShape([self.default_ventilation_action_spec_num_actions]))
        self.assertAllClose(distribution_step.action.loc, tf.constant([[0.4, 120.0], [0.2, 90.0]], dtype=tf.float32))


if __name__ == '__main__':
    tf.test.main()
