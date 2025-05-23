"""Factory function for creating Soft Actor-Critic (SAC) agents.

This module provides a utility function to instantiate a SAC agent from the
TF-Agents library, configured with specified actor and critic networks,
optimizers, and hyperparameters.
"""

from typing import Optional, Sequence

import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.typing import types

from smart_control.reinforcement_learning.agents.networks.sac_networks import create_sequential_actor_network
from smart_control.reinforcement_learning.agents.networks.sac_networks import create_sequential_critic_network


def create_sac_agent(
    time_step_spec: types.TimeStep,
    action_spec: types.NestedTensorSpec,
    # Actor network parameters
    actor_fc_layers: Sequence[int] = (256, 256),
    actor_network: Optional[network.Network] = None,
    # Critic network parameters
    critic_obs_fc_layers: Sequence[int] = (), # Default to identity if not specified
    critic_action_fc_layers: Sequence[int] = (), # Default to identity
    critic_joint_fc_layers: Sequence[int] = (256, 256), # Common default
    critic_network: Optional[network.Network] = None,
    # Optimizer parameters
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    alpha_learning_rate: float = 3e-4,
    # Agent hyperparameters
    gamma: float = 0.99,
    target_update_tau: float = 0.005,
    target_update_period: int = 1,
    reward_scale_factor: float = 1.0,
    # Training configuration
    gradient_clipping: Optional[float] = None,
    debug_summaries: bool = False,
    summarize_grads_and_vars: bool = False,
    train_step_counter: Optional[tf.Variable] = None,
) -> sac_agent.SacAgent:
  """Instantiates and configures a Soft Actor-Critic (SAC) agent.

  This function simplifies the creation of a SAC agent by providing sensible
  defaults and allowing customization of network architectures and training
  parameters. If custom actor or critic networks are not provided, default
  sequential networks will be created based on the `*_fc_layers` arguments.

  Args:
    time_step_spec (types.TimeStep): A `TimeStep` spec describing the
      observations from the environment.
    action_spec (types.NestedTensorSpec): A nest of `BoundedTensorSpec`
      representing the actions expected by the agent and output by the policy.
    actor_fc_layers (Sequence[int]): A sequence of integers defining the number
      of units in each fully connected layer of the default actor network.
      Used if `actor_network` is None.
    actor_network (Optional[network.Network]): A custom Keras network instance
      for the actor. If None, a default one is created.
    critic_obs_fc_layers (Sequence[int]): Sequence of units for dense layers
      processing observations in the default critic network.
    critic_action_fc_layers (Sequence[int]): Sequence of units for dense layers
      processing actions in the default critic network.
    critic_joint_fc_layers (Sequence[int]): Sequence of units for dense layers
      in the critic network after concatenating processed observations and
      actions.
    critic_network (Optional[network.Network]): A custom Keras network instance
      for the critic. If None, a default one is created.
    actor_learning_rate (float): Learning rate for the actor network's
      optimizer.
    critic_learning_rate (float): Learning rate for the critic network's
      optimizer.
    alpha_learning_rate (float): Learning rate for the entropy regularization
      parameter (alpha) optimizer.
    gamma (float): Discount factor for future rewards, in [0, 1].
    target_update_tau (float): Factor for soft updating target networks.
      A value of 1.0 means a hard update.
    target_update_period (int): Period (number of training steps) for updating
      the target networks.
    reward_scale_factor (float): Factor by which to scale rewards before use.
    gradient_clipping (Optional[float]): If not None, the L2 norm value to
      which gradients are clipped.
    debug_summaries (bool): If True, diagnostic summaries for debugging will be
      created.
    summarize_grads_and_vars (bool): If True, summaries of gradients and
      trainable variables will be created.
    train_step_counter (Optional[tf.Variable]): An optional `tf.Variable` to
      increment during training. Defaults to `tf.compat.v1.train.get_global_step()`.

  Returns:
    sac_agent.SacAgent: A configured instance of the TF-Agents SAC agent.

  Example:
    ```python
    # Assuming time_step_spec and action_spec are defined
    # (e.g., from a TF-Agents environment)
    # observation_spec = tensor_spec.TensorSpec((4,), tf.float32, 'obs')
    # time_step_spec = ts.time_step_spec(observation_spec)
    # action_spec = tensor_spec.BoundedTensorSpec(
    #     (1,), tf.float32, minimum=-1.0, maximum=1.0, name='action'
    # )
    #
    # sac_ag = create_sac_agent(
    #     time_step_spec=time_step_spec,
    #     action_spec=action_spec,
    #     actor_fc_layers=(128, 64),
    #     critic_joint_fc_layers=(128, 64)
    # )
    # print(sac_ag.name) # Output: SacAgent
    ```
  """
  if train_step_counter is None:
    train_step_counter = tf.compat.v1.train.get_or_create_global_step()

  if actor_network is None:
    actor_network = create_sequential_actor_network(
        actor_fc_layers=actor_fc_layers, action_tensor_spec=action_spec
    )
    if not actor_fc_layers: # Log if default identity network is used
        tf.compat.v1.logging.info(
            "Using identity network for SAC actor as actor_fc_layers is empty."
        )

  if critic_network is None:
    critic_network = create_sequential_critic_network(
        obs_fc_layer_units=critic_obs_fc_layers,
        action_fc_layer_units=critic_action_fc_layers,
        joint_fc_layer_units=critic_joint_fc_layers,
    )
    if not critic_obs_fc_layers and not critic_action_fc_layers and \
       not critic_joint_fc_layers:
        tf.compat.v1.logging.info(
            "Using effectively identity network for SAC critic as all "
            "critic layer units are empty."
        )

  # Instantiate the SAC agent from TF-Agents library
  agent = sac_agent.SacAgent(
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      actor_network=actor_network,
      critic_network=critic_network,
      actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_learning_rate),
      critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_learning_rate),
      alpha_optimizer=tf.keras.optimizers.Adam(learning_rate=alpha_learning_rate),
      target_update_tau=target_update_tau,
      target_update_period=target_update_period,
      td_errors_loss_fn=tf.math.squared_difference, # Standard SAC loss
      gamma=gamma,
      reward_scale_factor=reward_scale_factor,
      gradient_clipping=gradient_clipping,
      debug_summaries=debug_summaries,
      summarize_grads_and_vars=summarize_grads_and_vars,
      train_step_counter=train_step_counter,
  )
  return agent
