import unittest
from unittest.mock import patch, MagicMock

import tensorflow as tf
import numpy as np

from smart_control.reinforcement_learning.agents import sac_agent as sc_sac_agent
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step
from tf_agents.policies import actor_policy
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.agents.sac import sac_agent as tfa_sac_agent # For mocking its __init__
from tf_agents.utils import common


class SACAgentTest(tf.test.TestCase):

    def _create_dummy_specs(self):
        self.observation_spec = tensor_spec.TensorSpec(shape=(4,), dtype=tf.float32, name='observation')
        self.action_spec = tensor_spec.BoundedTensorSpec(shape=(1,), dtype=tf.float32, name='action', minimum=-1.0, maximum=1.0)
        self.time_step_spec = time_step.time_step_spec(self.observation_spec)
        return self.time_step_spec, self.action_spec

    def _create_dummy_actor_network(self, action_spec):
        return actor_distribution_network.ActorDistributionNetwork(
            self.observation_spec,
            action_spec,
            fc_layer_params=(10,),
            continuous_projection_network=normal_projection_network.NormalProjectionNetwork,
        )

    def _create_dummy_critic_network(self):
        # A simple critic network that takes observations and actions and returns a Q-value.
        # This is a very basic mock and might need adjustment based on actual SacAgent's critic req.
        input_tensors = (self.observation_spec, self.action_spec)
        return actor_distribution_network.ActorDistributionNetwork( # Using Actor for simplicity here, replace if specific critic needed
            input_tensors,
            tensor_spec.TensorSpec(shape=(1,), dtype=tf.float32, name='q_value'),
            fc_layer_params=(10,)
        )

    def setUp(self):
        super().setUp()
        tf.compat.v1.enable_v2_behavior() # Ensure TF2 behavior
        self.time_step_spec, self.action_spec = self._create_dummy_specs()
        self.actor_network = self._create_dummy_actor_network(self.action_spec)
        self.critic_network = self._create_dummy_critic_network() # Simplified
        self.actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
        self.alpha_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)

    @patch('tf_agents.agents.sac.sac_agent.SacAgent.__init__')
    def test_init(self, mock_super_init):
        agent = sc_sac_agent.SACAgent(
            self.time_step_spec,
            self.action_spec,
            actor_network=self.actor_network,
            critic_network=self.critic_network,
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer,
            alpha_optimizer=self.alpha_optimizer,
            target_update_tau=0.005,
            target_update_period=1,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=0.99,
            reward_scale_factor=1.0,
            train_step_counter=tf.Variable(0)
        )
        mock_super_init.assert_called_once()
        # Check if specific args passed to super were as expected
        args, kwargs = mock_super_init.call_args
        self.assertEqual(kwargs['time_step_spec'], self.time_step_spec)
        self.assertEqual(kwargs['action_spec'], self.action_spec)
        self.assertEqual(kwargs['actor_network'], self.actor_network)
        self.assertEqual(kwargs['critic_network'], self.critic_network)
        self.assertEqual(kwargs['actor_optimizer'], self.actor_optimizer)
        self.assertEqual(kwargs['critic_optimizer'], self.critic_optimizer)
        self.assertEqual(kwargs['alpha_optimizer'], self.alpha_optimizer)
        self.assertEqual(kwargs['target_update_tau'], 0.005)
        self.assertEqual(kwargs['target_update_period'], 1)
        self.assertEqual(kwargs['td_errors_loss_fn'], tf.math.squared_difference)
        self.assertEqual(kwargs['gamma'], 0.99)
        self.assertEqual(kwargs['reward_scale_factor'], 1.0)
        self.assertIsInstance(agent, sc_sac_agent.SACAgent)

    def test_experience_to_transitions_single(self):
        # Create single TimeStep and PolicyStep
        observation = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
        action = tf.constant([0.5], dtype=tf.float32)
        policy_info = {'log_probability': tf.constant([0.1], dtype=tf.float32)}

        current_time_step = time_step.restart(observation, batch_size=1)
        policy_step = trajectory.PolicyStep(action=action, state=(), info=policy_info)
        next_time_step = time_step.transition(observation + 1.0, reward=tf.constant([1.0], dtype=tf.float32), discount=tf.constant([1.0], dtype=tf.float32))

        # Convert to transitions
        transitions = sc_sac_agent.SACAgent._experience_to_transitions(
            (current_time_step, policy_step, next_time_step)
        )

        # Verify output is a single Trajectory
        self.assertIsInstance(transitions, trajectory.Trajectory)
        self.assertFalse(tensor_spec.is_batched(transitions.observation)) # Should not be batched

        # Verify fields
        self.assertAllEqual(transitions.step_type, current_time_step.step_type)
        self.assertAllClose(transitions.observation, current_time_step.observation)
        self.assertAllClose(transitions.action, policy_step.action)
        self.assertAllClose(transitions.policy_info['log_probability'], policy_info['log_probability'])
        self.assertAllEqual(transitions.next_step_type, next_time_step.step_type)
        self.assertAllClose(transitions.reward, next_time_step.reward)
        self.assertAllClose(transitions.discount, next_time_step.discount)

        # Check dtypes
        self.assertEqual(transitions.observation.dtype, tf.float32)
        self.assertEqual(transitions.action.dtype, tf.float32)
        self.assertEqual(transitions.reward.dtype, tf.float32)
        self.assertEqual(transitions.discount.dtype, tf.float32)
        self.assertEqual(transitions.step_type.dtype, tf.int32) # Default for step_type
        self.assertEqual(transitions.next_step_type.dtype, tf.int32) # Default for next_step_type

    def test_experience_to_transitions_batched(self):
        # Create two sets of TimeStep and PolicyStep
        obs1 = tf.constant([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
        act1 = tf.constant([0.5], dtype=tf.float32)
        p_info1 = {'log_probability': tf.constant([0.1], dtype=tf.float32)}
        ts1 = time_step.restart(obs1, batch_size=1)
        ps1 = trajectory.PolicyStep(action=act1, state=(), info=p_info1)
        next_ts1 = time_step.transition(obs1 + 1.0, reward=tf.constant([1.0], dtype=tf.float32), discount=tf.constant([1.0], dtype=tf.float32))

        obs2 = tf.constant([5.0, 6.0, 7.0, 8.0], dtype=tf.float32)
        act2 = tf.constant([-0.5], dtype=tf.float32)
        p_info2 = {'log_probability': tf.constant([0.2], dtype=tf.float32)}
        ts2 = time_step.transition(obs2, reward=tf.constant([0.0], dtype=tf.float32), discount=tf.constant([1.0], dtype=tf.float32)) # MID step
        ps2 = trajectory.PolicyStep(action=act2, state=(), info=p_info2)
        next_ts2 = time_step.termination(obs2 + 1.0, reward=tf.constant([2.0], dtype=tf.float32)) # LAST step

        # Manually batch them
        # Note: For tf_agents TimeStep, direct stacking of the objects is not straightforward
        # because of the way spec checks work. We need to stack the components.
        batched_current_time_step = time_step.TimeStep(
            step_type=tf.stack([ts1.step_type[0], ts2.step_type[0]]), # Extract scalar from batch_size 1
            reward=tf.stack([ts1.reward[0], ts2.reward[0]]),
            discount=tf.stack([ts1.discount[0], ts2.discount[0]]),
            observation=tf.stack([ts1.observation[0], ts2.observation[0]])
        )
        # For PolicyStep, action and info can be stacked directly if they are already batched or can be made so
        batched_policy_step = trajectory.PolicyStep(
            action=tf.stack([ps1.action[0], ps2.action[0]]),
            state=(), # Assuming empty state tuple, otherwise needs careful handling
            info={'log_probability': tf.stack([p_info1['log_probability'][0], p_info2['log_probability'][0]])}
        )
        batched_next_time_step = time_step.TimeStep(
            step_type=tf.stack([next_ts1.step_type[0], next_ts2.step_type[0]]),
            reward=tf.stack([next_ts1.reward[0], next_ts2.reward[0]]),
            discount=tf.stack([next_ts1.discount[0], next_ts2.discount[0]]),
            observation=tf.stack([next_ts1.observation[0], next_ts2.observation[0]])
        )

        experience_batch = (batched_current_time_step, batched_policy_step, batched_next_time_step)

        # Convert to transitions
        transitions = sc_sac_agent.SACAgent._experience_to_transitions(experience_batch)

        # Verify output is a batched Trajectory
        self.assertIsInstance(transitions, trajectory.Trajectory)
        self.assertTrue(tensor_spec.is_batched(transitions.observation))
        self.assertEqual(tf.shape(transitions.observation)[0], 2) # Batch size of 2

        # Verify fields for the batch
        self.assertAllEqual(transitions.step_type, batched_current_time_step.step_type)
        self.assertAllClose(transitions.observation, batched_current_time_step.observation)
        self.assertAllClose(transitions.action, batched_policy_step.action)
        self.assertAllClose(transitions.policy_info['log_probability'], batched_policy_step.info['log_probability'])
        self.assertAllEqual(transitions.next_step_type, batched_next_time_step.step_type)
        self.assertAllClose(transitions.reward, batched_next_time_step.reward)
        self.assertAllClose(transitions.discount, batched_next_time_step.discount)

        # Check dtypes
        self.assertEqual(transitions.observation.dtype, tf.float32)
        self.assertEqual(transitions.action.dtype, tf.float32)
        self.assertEqual(transitions.reward.dtype, tf.float32)
        self.assertEqual(transitions.discount.dtype, tf.float32)
        self.assertEqual(transitions.step_type.dtype, tf.int32)
        self.assertEqual(transitions.next_step_type.dtype, tf.int32)


if __name__ == '__main__':
    unittest.main()
