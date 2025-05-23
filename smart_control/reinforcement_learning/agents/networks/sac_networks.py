"""Defines actor and critic network architectures for SAC agents.

This module provides factory functions to construct the neural networks
required by the Soft Actor-Critic (SAC) agent. These include:
- Fully connected networks as building blocks.
- Critic networks that process observations and actions to estimate Q-values.
- Actor networks that output a distribution over actions given an observation.

The networks are built using TensorFlow Keras and are compatible with TF-Agents.
"""

import functools
from typing import Any, Mapping, Sequence, Tuple

import tensorflow as tf
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.keras_layers import inner_reshape
from tf_agents.networks import nest_map
from tf_agents.networks import sequential
from tf_agents.typing import types

# Default Keras dense layer with ReLU activation and Glorot uniform initializer.
dense = functools.partial(
    tf.keras.layers.Dense,
    activation=tf.keras.activations.relu,
    kernel_initializer="glorot_uniform",
)


def create_fc_network(layer_units: Sequence[int]) -> tf.keras.Model:
  """Creates a Keras Sequential model of fully connected layers.

  Each layer uses the default `dense` configuration (ReLU activation,
  Glorot uniform kernel initializer).

  Args:
    layer_units (Sequence[int]): A list or tuple where each element is the
      number of units in a dense layer.

  Returns:
    tf.keras.Model: A Keras Sequential model composed of the specified dense
    layers.

  Example:
    >>> fc_net = create_fc_network([256, 128])
    >>> print(fc_net.summary()) # Output depends on input shape if built
  """
  return sequential.Sequential(
      [dense(num_units) for num_units in layer_units], name="fc_network"
  )


def create_identity_layer() -> tf.keras.layers.Layer:
  """Creates a Keras Lambda layer that returns its input unchanged.

  This can be used as a placeholder in network constructions where a
  sub-network is optional.

  Returns:
    tf.keras.layers.Layer: An identity Keras layer.
  """
  return tf.keras.layers.Lambda(lambda x: x, name="identity_layer")


def create_sequential_critic_network(
    obs_fc_layer_units: Sequence[int],
    action_fc_layer_units: Sequence[int],
    joint_fc_layer_units: Sequence[int],
) -> sequential.Sequential:
  """Creates a sequential critic network for a SAC agent.

  The critic network takes observations and actions as input and outputs a
  Q-value. It typically consists of:
  1. Separate fully connected networks to process observations and actions.
  2. A concatenation of the processed observations and actions.
  3. A joint fully connected network to combine the concatenated features.
  4. A final dense layer to output the scalar Q-value.

  Args:
    obs_fc_layer_units (Sequence[int]): Number of units in each dense layer
      for processing observations. If empty, an identity layer is used.
    action_fc_layer_units (Sequence[int]): Number of units in each dense layer
      for processing actions. If empty, an identity layer is used.
    joint_fc_layer_units (Sequence[int]): Number of units in each dense layer
      of the joint network after concatenation. If empty, an identity layer is
      used.

  Returns:
    sequential.Sequential: A Keras Sequential model representing the critic.
      The model expects a tuple/list of (observation, action) as input.
  """

  def split_inputs(
      inputs: Tuple[types.Tensor, types.Tensor]
  ) -> Mapping[str, types.Tensor]:
    """Splits the input tuple into a dictionary of observation and action."""
    return {"observation": inputs[0], "action": inputs[1]}

  obs_network = (
      create_fc_network(obs_fc_layer_units)
      if obs_fc_layer_units
      else create_identity_layer()
  )
  action_network = (
      create_fc_network(action_fc_layer_units)
      if action_fc_layer_units
      else create_identity_layer()
  )
  joint_network = (
      create_fc_network(joint_fc_layer_units)
      if joint_fc_layer_units
      else create_identity_layer()
  )

  # Output layer for the Q-value.
  value_layer = tf.keras.layers.Dense(
      1, activation=None, kernel_initializer="glorot_uniform"
  )

  return sequential.Sequential(
      [
          tf.keras.layers.Lambda(split_inputs),
          nest_map.NestMap({"observation": obs_network, "action": action_network}),
          nest_map.NestFlatten(), # Flattens the dict to a list of tensors
          tf.keras.layers.Concatenate(),
          joint_network,
          value_layer,
          # Reshape output from [batch_size, 1] to [batch_size]
          inner_reshape.InnerReshape(current_shape=[1], new_shape=[]),
      ],
      name="sequential_critic",
  )


class _TanhNormalProjectionNetworkWrapper(
    tanh_normal_projection_network.TanhNormalProjectionNetwork
):
  """Wraps TanhNormalProjectionNetwork to set a predefined `outer_rank`.

  The standard `TanhNormalProjectionNetwork` infers `outer_rank` from the
  input's shape. This wrapper allows explicitly setting `outer_rank`, which
  can be useful when the input shape might be ambiguous or when a specific
  rank is required by the subsequent network structure, especially when used
  within a `tf.keras.Sequential` model where `call` arguments might be limited.

  Attributes:
    predefined_outer_rank (int): The fixed outer rank to be used in the call
      to the parent class's `call` method.
  """

  def __init__(self, sample_spec: types.NestedTensorSpec, predefined_outer_rank: int = 1):
    """Initializes the wrapper.

    Args:
      sample_spec (types.NestedTensorSpec): A nest of BoundedTensorSpec
        defining the shape and bounds of the actions to be projected.
      predefined_outer_rank (int): The outer rank to be passed to the
        underlying projection network's call method. Defaults to 1.
    """
    super().__init__(sample_spec)
    self.predefined_outer_rank = predefined_outer_rank

  def call(self, inputs: types.Tensor, **kwargs: Any) -> types.Distribution:
    """Calls the underlying TanhNormalProjectionNetwork with predefined outer_rank.

    It also removes 'step_type' from kwargs if present, as it's not expected
    by the parent `call` method when used in certain TF-Agents contexts.

    Args:
      inputs (types.Tensor): The input tensor to the projection network.
      **kwargs (Any): Additional keyword arguments.

    Returns:
      types.Distribution: A nest of tfp.distributions.Distribution representing
      the projected action distribution.
    """
    kwargs["outer_rank"] = self.predefined_outer_rank
    # step_type is sometimes passed by TF-Agents internals but not expected
    # by the TanhNormalProjectionNetwork's call method directly.
    if "step_type" in kwargs:
      del kwargs["step_type"]
    return super().call(inputs, **kwargs)


def create_sequential_actor_network(
    actor_fc_layers: Sequence[int],
    action_tensor_spec: types.NestedTensorSpec,
) -> sequential.Sequential:
  """Creates a sequential actor network for a SAC agent.

  The actor network takes observations as input and outputs parameters for a
  distribution over actions (typically a Tanh-squashed Normal distribution
  for SAC). It usually consists of:
  1. A series of fully connected layers to process the observation.
  2. A projection layer that maps the processed observation to the parameters
     of the action distribution (e.g., mean and standard deviation).

  Args:
    actor_fc_layers (Sequence[int]): Number of units in each dense layer of
      the main body of the actor network.
    action_tensor_spec (types.NestedTensorSpec): A nest of BoundedTensorSpec
      describing the shape, dtype, and bounds of the actions. This is used by
      the TanhNormalProjectionNetwork.

  Returns:
    sequential.Sequential: A Keras Sequential model representing the actor.
      The model expects an observation tensor as input.
  """

  def tile_as_nest(non_nested_output: types.Tensor) -> types.NestedTensor:
    """Tiles the non-nested output to match the structure of action_tensor_spec.

    This is necessary if the projection network expects inputs structured like
    the action spec (e.g., if action_tensor_spec is a nest).
    """
    return tf.nest.map_structure(lambda _: non_nested_output, action_tensor_spec)

  # Main body of the actor network
  fc_net = [dense(num_units) for num_units in actor_fc_layers]

  # Projection network for SAC (outputs parameters for TanhNormal distribution)
  # We use a NestMap to apply the projection network to each element if
  # action_tensor_spec is a nest. If it's a single tensor spec, NestMap
  # still works.
  projection_layers = [
      nest_map.NestMap(
          tf.nest.map_structure(
              _TanhNormalProjectionNetworkWrapper, action_tensor_spec
          )
      )
  ]

  # If action_tensor_spec is a nest, the output of fc_net needs to be
  # tiled to match this structure before passing to NestMap.
  # If action_tensor_spec is not a nest, Lambda(tile_as_nest) might be
  # slightly redundant but harmless if NestMap handles single tensor inputs
  # correctly.
  # The original code had Lambda(tile_as_nest) after fc_net.
  # This implies fc_net output is a single tensor, and it needs to be
  # structured like action_tensor_spec for the NestMap.
  return sequential.Sequential(
      fc_net + [tf.keras.layers.Lambda(tile_as_nest)] + projection_layers,
      name="sequential_actor",
  )
