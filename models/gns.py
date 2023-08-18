import numpy as np
import yaml
import tensorflow as tf
from os.path import join, exists, dirname, abspath
from abc import ABC, abstractmethod

from o3d.utils import Config
from utils.tools.losses import get_loss
from utils.tools.neighbor import neighbors_mask, reduce_subarrays_sum_multi

from .base_model import BaseModel
import graph_nets as gn
import sonnet as snt

STD_EPSILON = 1e-8
from typing import Callable

Reducer = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]
import functools
from sklearn import neighbors

import collections

Stats = collections.namedtuple('Stats', ['mean', 'std'])


def _combine_std(std_x, std_y):
    return np.sqrt(std_x ** 2 + std_y ** 2)


import tensorflow as tf


def merge_dims(tensor, start, size):
    """Merges dimensions of a tensor from 'start' for 'size' dimensions."""
    static_input_shape = tensor.shape.as_list()
    rank = len(static_input_shape)

    if start < 0:
        start += rank  # Negative indexing

    if rank < start + size:
        raise ValueError(f"Rank of inputs must be at least {start + size}.")

    initial = static_input_shape[:start]
    middle = static_input_shape[start:start + size]
    final = static_input_shape[start + size:]

    if None in middle:
        middle = [None]
    else:
        middle = [tf.reduce_prod(middle)]

    new_shape = initial + middle + final

    return tf.reshape(tensor, new_shape)


class GNS(BaseModel):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(self,
                 name,
                 timestep,
                 particle_radii,
                 grav,
                 transformation={},
                 window_dens=None,
                 num_dimensions=3,
                 connectivity_radius=0.035,
                 latent_size=128,
                 hidden_size=128,
                 hidden_layers=2,
                 message_passing_steps=10,
                 num_particle_types=2,
                 particle_type_embedding_size=16,
                 loss={
                     "weighted_mse": {
                         "typ": "weighted_mse",
                         "fac": 1.0,
                         "gamma": 0.25,
                         "neighbor_scale": 0.025
                     }
                 },
                 **kwargs):
        super().__init__(name=name,
                         timestep=timestep,
                         particle_radii=particle_radii,
                         transformation=transformation,
                         grav=grav,
                         window_dens=window_dens,
                         **kwargs)
        model_kwargs = dict(
            latent_size=latent_size,
            mlp_hidden_size=hidden_size,
            mlp_num_hidden_layers=hidden_layers,
            num_message_passing_steps=message_passing_steps)

        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        # loss setup
        self.loss_fn = {}
        for l, v in loss.items():
            if v["typ"] == "dense":
                if not "radius" in v:
                    v["radius"] = connectivity_radius
            self.loss_fn[l] = get_loss(**v)

        self._graph_network = EncodeProcessDecode(
            output_size=num_dimensions, **model_kwargs)

        if self._num_particle_types > 1:
            self._particle_type_embedding = tf.Variable(
                tf.random.normal([self._num_particle_types, particle_type_embedding_size]),
                trainable=True)

    def forward(self, prev, data, training=True, **kwargs):
        position_sequence, n_particles_per_example, _, particle_types = prev
        return self._build(position_sequence, n_particles_per_example, particle_types=particle_types)

    def preprocess(self, data, training=True, **kwargs):
        _pos, _vel, acc, feats, box, bfeats = data
        # 在此处插入您的数据预处理代码
        position_sequence = tf.concat([_pos, box], axis=0)[:, None, :]
        n_particles_per_example = [tf.shape(position_sequence)[0]]
        particle_types = tf.concat([tf.zeros_like(_pos[:, 0], dtype=tf.int32), tf.ones_like(box[:, 0], dtype=tf.int32)],
                                   axis=0)
        return [position_sequence, n_particles_per_example, None, particle_types]

    def postprocess(self, prev, data, training=True, **kwargs):
        _pos, _vel, acc, feats, box, bfeats = data
        # # Use an Euler integrator to go from acceleration to position, assuming
        # # a dt=1 corresponding to the size of the finite difference.
        last_position = _pos
        last_delta_position = _vel * self.timestep

        delta_position = last_delta_position + prev[:tf.shape(_pos)[0]]  # * dt = 1
        new_position = last_position + delta_position  # * dt = 1

        # 在此处插入您的数据后处理代码
        self.pos_correction = new_position - last_position
        vel = (self.pos_correction) / self.timestep
        return new_position, vel

    # 其他函数如_build, _encoder_preprocessor, _decoder_postprocessor等保持不变
    def _build(self, position_sequence, n_particles_per_example,
               global_context=None, particle_types=None):
        """Produces a model step, outputting the next position for each particle.

        Args:
          position_sequence: Sequence of positions for each node in the batch,
            with shape [num_particles_in_batch, sequence_length, num_dimensions]
          n_particles_per_example: Number of particles for each graph in the batch
            with shape [batch_size]
          global_context: Tensor of shape [batch_size, context_size], with global
            context.
          particle_types: Integer tensor of shape [num_particles_in_batch] with
            the integer types of the particles, from 0 to `num_particle_types - 1`.
            If None, we assume all particles are the same type.

        Returns:
          Next position with shape [num_particles_in_batch, num_dimensions] for one
          step into the future from the input sequence.
        """
        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence, n_particles_per_example, global_context,
            particle_types)

        normalized_acceleration = self._graph_network(input_graphs_tuple)

        next_position = self._decoder_postprocessor(
            normalized_acceleration, position_sequence)

        return next_position

    def _encoder_preprocessor(
            self, position_sequence, n_node, global_context, particle_types):
        # Extract important features from the position_sequence.
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = time_diff(position_sequence)  # Finite-difference.

        # Get connectivity of the graph.
        (senders, receivers, n_edge
         ) = compute_connectivity_for_batch_pyfunc(
            most_recent_position, n_node, self._connectivity_radius)

        # Collect node features.
        node_features = []

        # Normalized velocity sequence, merging spatial an time axis.
        normalized_velocity_sequence = velocity_sequence

        flat_velocity_sequence = merge_dims(normalized_velocity_sequence, start=1, size=2)

        node_features.append(flat_velocity_sequence)

        # Particle type.
        if self._num_particle_types > 1:
            particle_type_embeddings = tf.nn.embedding_lookup(
                self._particle_type_embedding, particle_types)
            node_features.append(particle_type_embeddings)

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        normalized_relative_displacements = (
                                                    tf.gather(most_recent_position, senders) -
                                                    tf.gather(most_recent_position,
                                                              receivers)) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = tf.norm(
            normalized_relative_displacements, axis=-1, keepdims=True)
        edge_features.append(normalized_relative_distances)

        return gn.graphs.GraphsTuple(
            nodes=tf.concat(node_features, axis=-1),
            edges=tf.concat(edge_features, axis=-1),
            globals=global_context,  # self._graph_net will appending this to nodes.
            n_node=n_node,
            n_edge=n_edge,
            senders=senders,
            receivers=receivers,
        )

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        # The model produces the output in normalized space so we apply inverse
        # normalization.
        acceleration = normalized_acceleration

        return acceleration

    def loss_keys(self):
        return self.loss_fn.keys()

    def loss(self, results, data):
        loss = {}

        pred = results[0]
        target = data[1]

        for n, l in self.loss_fn.items():
            loss[n] = l(target,
                        pred,
                        input=data[0],
                        target_prev=data[2],
                        pre_steps=data[3],
                        pos_correction=self.pos_correction)

        return loss


def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]


def build_mlp(
        hidden_size: int, num_hidden_layers: int, output_size: int) -> snt.Module:
    """Builds an MLP."""
    return snt.nets.MLP(
        output_sizes=[hidden_size] * num_hidden_layers + [
            output_size])  # pytype: disable=bad-return-type  # gen-stub-imports


class EncodeProcessDecode(snt.Module):
    """Encode-Process-Decode function approximator for learnable simulator."""

    def __init__(
            self,
            latent_size: int,
            mlp_hidden_size: int,
            mlp_num_hidden_layers: int,
            num_message_passing_steps: int,
            output_size: int,
            reducer: Reducer = tf.math.unsorted_segment_sum,
            name: str = "EncodeProcessDecode"):
        """Inits the model.

        Args:
          latent_size: Size of the node and edge latent representations.
          mlp_hidden_size: Hidden layer size for all MLPs.
          mlp_num_hidden_layers: Number of hidden layers in all MLPs.
          num_message_passing_steps: Number of message passing steps.
          output_size: Output size of the decode node representations as required
            by the downstream update function.
          reducer: Reduction to be used when aggregating the edges in the nodes in
            the interaction network. This should be a callable whose signature
            matches tf.math.unsorted_segment_sum.
          name: Name of the model.
        """

        super().__init__(name=name)

        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self._reducer = reducer

        self._networks_builder()

    def __call__(self, input_graph: gn.graphs.GraphsTuple, *args, **kwargs) -> tf.Tensor:
        """Forward pass of the learnable dynamics model."""

        # Encode the input_graph.
        latent_graph_0 = self._encode(input_graph)

        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)

        # Decode from the last latent graph.
        return self._decode(latent_graph_m)

    def _networks_builder(self):
        """Builds the networks."""

        def build_mlp_with_layer_norm():
            mlp = build_mlp(
                hidden_size=self._mlp_hidden_size,
                num_hidden_layers=self._mlp_num_hidden_layers,
                output_size=self._latent_size)
            return snt.Sequential([mlp, snt.LayerNorm(axis=slice(1, None), create_scale=True, create_offset=True)])

        # The encoder graph network independently encodes edge and node features.
        encoder_kwargs = dict(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm)
        self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)

        # Create `num_message_passing_steps` graph networks with unshared parameters
        # that update the node and edge latent features.
        # Note that we can use `modules.InteractionNetwork` because
        # it also outputs the messages as updated edge latent features.
        self._processor_networks = []
        for _ in range(self._num_message_passing_steps):
            self._processor_networks.append(
                gn.modules.InteractionNetwork(
                    edge_model_fn=build_mlp_with_layer_norm,
                    node_model_fn=build_mlp_with_layer_norm,
                    reducer=self._reducer))

        # The decoder MLP decodes node latent features into the output size.
        self._decoder_network = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size)

    def _encode(
            self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        """Encodes the input graph features into a latent graph."""

        # Copy the globals to all of the nodes, if applicable.
        if input_graph.globals is not None:
            broadcasted_globals = gn.blocks.broadcast_globals_to_nodes(input_graph)
            input_graph = input_graph.replace(
                nodes=tf.concat([input_graph.nodes, broadcasted_globals], axis=-1),
                globals=None)

        # Encode the node and edge features.
        latent_graph_0 = self._encoder_network(input_graph)
        return latent_graph_0

    def _process(
            self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        """Processes the latent graph with several steps of message passing."""

        # Do `m` message passing steps in the latent graphs.
        # (In the shared parameters case, just reuse the same `processor_network`)
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        for k in range(len(self._processor_networks)):
            processor_network_k = self._processor_networks[k]
            latent_graph_k = self._process_step(
                processor_network_k, latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k

        latent_graph_m = latent_graph_k
        return latent_graph_m

    def _process_step(
            self, processor_network_k: snt.Module,
            latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        """Single step of message passing with node/edge residual connections."""

        # One step of message passing.
        latent_graph_k = processor_network_k(latent_graph_prev_k)

        # Add residuals.
        latent_graph_k = latent_graph_k.replace(
            nodes=latent_graph_k.nodes + latent_graph_prev_k.nodes,
            edges=latent_graph_k.edges + latent_graph_prev_k.edges)
        return latent_graph_k

    def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
        """Decodes from the latent graph."""
        return self._decoder_network(latent_graph.nodes)


def compute_connectivity_for_batch_pyfunc(
        positions, n_node, radius, add_self_edges=True):
    """`_compute_connectivity_for_batch` wrapped in a pyfunc."""
    partial_fn = functools.partial(
        _compute_connectivity_for_batch, add_self_edges=add_self_edges)
    senders, receivers, n_edge = tf.py_function(
        partial_fn,
        [positions, n_node, radius],
        [tf.int32, tf.int32, tf.int32])
    senders.set_shape([None])
    receivers.set_shape([None])
    n_edge.set_shape(tf.shape(n_node))
    return senders, receivers, n_edge


def _compute_connectivity_for_batch(
        positions, n_node, radius, add_self_edges):
    """`compute_connectivity` for a batch of graphs.

    Args:
      positions: Positions of nodes in the batch of graphs. Shape:
        [num_nodes_in_batch, num_dims].
      n_node: Number of nodes for each graph in the batch. Shape:
        [num_graphs in batch].
      radius: Radius of connectivity.
      add_self_edges: Whether to include self edges or not.

    Returns:
      senders indices [num_edges_in_batch]
      receiver indices [num_edges_in_batch]
      number of edges per graph [num_graphs_in_batch]

    """

    # TODO(alvarosg): Consider if we want to support batches here or not.
    # Separate the positions corresponding to particles in different graphs.
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
    receivers_list = []
    senders_list = []
    n_edge_list = []
    num_nodes_in_previous_graphs = 0

    # Compute connectivity for each graph in the batch.
    for positions_graph_i in positions_per_graph_list:
        senders_graph_i, receivers_graph_i = _compute_connectivity(
            positions_graph_i, radius, add_self_edges)

        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        # Because the inputs will be concatenated, we need to add offsets to the
        # sender and receiver indices according to the number of nodes in previous
        # graphs in the same batch.
        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

        num_nodes_graph_i = len(positions_graph_i)
        num_nodes_in_previous_graphs += num_nodes_graph_i

    # Concatenate all of the results.
    senders = np.concatenate(senders_list, axis=0).astype(np.int32)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
    n_edge = np.stack(n_edge_list).astype(np.int32)

    return senders, receivers, n_edge


def _compute_connectivity(positions, radius, add_self_edges):
    """Get the indices of connected edges with radius connectivity.

    Args:
      positions: Positions of nodes in the graph. Shape:
        [num_nodes_in_graph, num_dims].
      radius: Radius of connectivity.
      add_self_edges: Whether to include self edges or not.

    Returns:
      senders indices [num_edges_in_graph]
      receiver indices [num_edges_in_graph]

    """
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        # Remove self edges.
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return senders, receivers
