"""Factory function for creating Soft Actor-Critic (SAC) agents.

This module provides a utility function, `create_sac_agent`, for instantiating
and configuring a `tf_agents.agents.sac.sac_agent.SacAgent`. This agent is
well-suited for continuous control tasks, often encountered in smart building
applications, due to its off-policy nature and exploration capabilities driven
by maximum entropy reinforcement learning.
"""

from typing import Optional, Sequence

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts

from smart_control.reinforcement_learning.agents.networks.sac_networks import create_sequential_actor_network
from smart_control.reinforcement_learning.agents.networks.sac_networks import create_sequential_critic_network


def create_sac_agent(
    time_step_spec: ts.TimeStepSpec,
    action_spec: tensor_spec.BoundedTensorSpec,
    # Actor network parameters
    actor_fc_layers: Sequence[int] = (256, 256),
    actor_network: Optional[network.Network] = None,
    # Critic network parameters
    critic_obs_fc_layers: Sequence[int] = (256, 128),
    critic_action_fc_layers: Sequence[int] = (256, 128),
    critic_joint_fc_layers: Sequence[int] = (256, 128),
    critic_network: Optional[network.Network] = None,
    # Optimizer parameters
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    alpha_learning_rate: float = 3e-4,
    # Agent parameters
    gamma: float = 0.99,
    target_update_tau: float = 0.005,
    target_update_period: int = 1,
    reward_scale_factor: float = 1.0,
    # Training parameters
    gradient_clipping: Optional[float] = None,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    train_step_counter: Optional[tf.Variable] = None,
) -> sac_agent.SacAgent:
  """Creates and configures a Soft Actor-Critic (SAC) agent.

  This function simplifies the instantiation of a `tf_agents.agents.sac.sac_agent.SacAgent`.
  It allows for customization of actor and critic networks, learning rates,
  and other SAC-specific hyperparameters. If custom networks are not provided,
  it creates default sequential networks using the specified layer configurations.

  Args:
    time_step_spec: A `tf_agents.trajectories.time_step.TimeStepSpec` defining the
      expected shape and type of observations from the environment.
    action_spec: A `tf_agents.specs.tensor_spec.BoundedTensorSpec` defining the
      shape, type, and bounds of actions expected by the environment.
    actor_fc_layers: A sequence of integers representing the number of units in
      each fully connected layer of the default actor network. Used if
      `actor_network` is not provided.
    actor_network: An optional `tf_agents.networks.network.Network` instance to
      use as the actor network. If `None`, a default network is created using
      `actor_fc_layers`.
    critic_obs_fc_layers: A sequence of integers for the observation pathway
      fully connected layers in the default critic network. Used if
      `critic_network` is not provided.
    critic_action_fc_layers: A sequence of integers for the action pathway
      fully connected layers in the default critic network. Used if
      `critic_network` is not provided.
    critic_joint_fc_layers: A sequence of integers for the joint pathway
      fully connected layers (after concatenating observation and action
      embeddings) in the default critic network. Used if `critic_network` is
      not provided.
    critic_network: An optional `tf_agents.networks.network.Network` instance to
      use as the critic network. If `None`, a default network is created using
      the `critic_*_fc_layers` arguments.
    actor_learning_rate: The learning rate for the actor network's optimizer.
    critic_learning_rate: The learning rate for the critic network's optimizer.
    alpha_learning_rate: The learning rate for the entropy regularization
      parameter (alpha) optimizer.
    gamma: The discount factor for future rewards, typically between 0 and 1.
    target_update_tau: The factor for soft updates of the target networks'
      weights (polyak averaging). A value of 1.0 means hard updates.
    target_update_period: The frequency (in training steps) at which to update
      the target networks.
    reward_scale_factor: A factor by which to scale rewards before they are used
      for training. This can help stabilize training.
    gradient_clipping: An optional float value. If provided, gradients will be
      clipped by this norm during training to prevent excessively large updates.
    debug_summaries: If True, diagnostic summaries (e.g., for network
      activations) will be generated for use with TensorBoard.
    summarize_grads_and_vars: If True, summaries of gradients and trainable
      variables will be generated for TensorBoard.
    train_step_counter: An optional `tf.Variable` to increment with each training
      step. If `None`, a new counter is created.

  Returns:
    An instance of `tf_agents.agents.sac.sac_agent.SacAgent` configured with
    the specified parameters.
  """
  # Create train step counter if not provided
  if train_step_counter is None:
    train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64)

  # Create networks if not provided
  if actor_network is None:
    actor_network = create_sequential_actor_network(
        actor_fc_layers=actor_fc_layers, action_tensor_spec=action_spec
    )

  if critic_network is None:
    critic_network = create_sequential_critic_network(
        obs_fc_layer_units=critic_obs_fc_layers,
        action_fc_layer_units=critic_action_fc_layers,
        joint_fc_layer_units=critic_joint_fc_layers,
    )

  # Create agent
  # Instantiate the SAC agent from TF-Agents library
  tf_agent_instance = sac_agent.SacAgent(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network,
      critic_network=critic_network,
      actor_optimizer=tf.keras.optimizers.Adam(
          learning_rate=actor_learning_rate
      ),
      critic_optimizer=tf.keras.optimizers.Adam(
          learning_rate=critic_learning_rate
      ),
      alpha_optimizer=tf.keras.optimizers.Adam(
          learning_rate=alpha_learning_rate
      ),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.math.squared_difference, # Standard loss for SAC
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step_counter,
  )

  return tf_agent_instance
