import tensorflow as tf
from scipy.spatial import cKDTree
import open3d.ml.tf as o3dml
import numpy as np
from utils.tools.losses import get_window_func, get_dilated_pos, compute_density, compute_pressure, get_loss, \
    compute_density_with_box

from .base_model import BaseModel

relu = tf.keras.activations.relu


def reduce_subarrays_sum_multi(values, row_splits):
    segment_ids = tf.ragged.row_splits_to_segment_ids(row_splits)
    sum = tf.math.unsorted_segment_sum(values, segment_ids, num_segments=tf.shape(row_splits)[0] - 1)
    return sum


class PolarConv(tf.keras.layers.Layer):
    def __init__(self, mlp_dims, activation=None, **kwargs):
        super(PolarConv, self).__init__(**kwargs)
        self.mlp_dims = mlp_dims
        self.activation = activation

    def build(self, input_shape):
        self.in_dim = input_shape[-1]
        self.mlp = tf.keras.models.Sequential()
        self.mlp.add(tf.keras.layers.Input(shape=(4,)))
        for dim in self.mlp_dims[:-1]:
            self.mlp.add(tf.keras.layers.Dense(dim, activation='relu'))
        self.mlp.add(tf.keras.layers.Dense(self.in_dim * self.mlp_dims[-1]))

    def call(self, feats, xyz, nns, query=None):
        if query is None:
            query = xyz
        neighbors_index, neighbors_row_splits, neighbors_distance = nns
        neighbors_index = tf.cast(neighbors_index, tf.int32)
        # 获取每个query点的邻居数
        neighbors_counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        # 对query进行扩展以匹配neighbors_index的形状
        expanded_query = tf.repeat(query, neighbors_counts, axis=0)
        # 确保diff的维度和xyz对应
        diff = tf.gather(xyz, neighbors_index) - expanded_query
        neighbors_distance = tf.stop_gradient(neighbors_distance)
        r = tf.sqrt(neighbors_distance + 1e-7)
        sin_theta = diff[..., 0] / r
        cos_theta = diff[..., 2] / r
        cos_phi = diff[..., 1] / r
        # Weighted feature
        polar_feats = tf.stack([r, sin_theta, cos_theta, cos_phi], axis=-1)
        weighted_feats = self.mlp(polar_feats)
        weighted_feats = tf.reshape(weighted_feats, shape=(-1, self.in_dim, self.mlp_dims[-1]))
        # Gather the features of the neighbors and apply the weights
        neighbor_feats = tf.gather(feats, neighbors_index)
        neighbor_feats = tf.einsum('ij,ijk->ik', neighbor_feats, weighted_feats)
        # Sum over neighbors
        new_feats = reduce_subarrays_sum_multi(neighbor_feats, tf.cast(neighbors_row_splits, tf.int32))
        if self.activation:
            new_feats = self.activation(new_feats)
        return new_feats

    def get_config(self):
        config = super().get_config()
        config.update({"mlp_dims": self.mlp_dims})
        return config


class PolarNet(BaseModel):
    def __init__(self,
                 name="PolarNet",
                 encoder_channels=[128, 128, 512],
                 decoder_channels=[512, 128, 3],
                 particle_radii=0.025,
                 query_radii=None,
                 dens_radius=None,
                 timestep=0.02,
                 grav=-9.81,
                 out_scale=[0.01, 0.01, 0.01],
                 rest_dens=8.0,
                 viscosity=0.02,
                 use_acc=False,
                 use_vel=True,
                 use_feats=False,
                 use_box_feats=True,
                 transformation={},
                 loss={
                     "weighted_mse": {
                         "typ": "weighted_mse",
                         "fac": 128.0,
                         "gamma": 0.5,
                         "neighbor_scale": 0.025
                     }
                 },
                 dens_feats=False,
                 pres_feats=False,
                 stiffness=20.0,
                 **kwargs):

        super().__init__(name='elevate',
                         timestep=timestep,
                         particle_radii=particle_radii,
                         transformation=transformation,
                         grav=grav,
                         **kwargs)
        self.query_radii = particle_radii * 4 if query_radii is None else query_radii
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels

        self.use_vel = use_vel
        self.use_acc = use_acc
        self.use_feats = use_feats
        self.use_box_feats = use_box_feats
        self.dens_feats = dens_feats
        self.pres_feats = pres_feats
        self.stiffness = stiffness
        self.rest_dens = rest_dens
        self.viscosity = viscosity
        if dens_radius is None:
            dens_radius = particle_radii
        self.dens_radius = dens_radius
        self.out_scale = tf.constant(out_scale)
        self.loss_fn = {}
        for l, v in loss.items():
            if v["typ"] == "dense":
                if not "radius" in v:
                    v["radius"] = query_radii
            self.loss_fn[l] = get_loss(**v)

        diameter = 2.0 * particle_radii
        volume = diameter * diameter * diameter * 0.8
        self.fluid_mass = volume * self.rest_dens
        self.box_masses, self.box = None, None
        self.radius_search = o3dml.layers.FixedRadiusSearch(ignore_query_point=True, return_distances=True)
        # self.w_f = self.build_w_f(self.query_radii, self.particle_radii)
        # self.w_b = 10000 * self.w_f
        self.fluid_encoder = PolarConv(mlp_dims=self.encoder_channels, name='fluid_encoder')
        self.solid_encoder = PolarConv(mlp_dims=self.encoder_channels, name='solid_encoder')
        self.mlps = []
        for i, dim in enumerate(self.decoder_channels):
            activation = 'relu' if i == len(self.decoder_channels) - 1 else None
            mlp = tf.keras.layers.Dense(dim, name='mlp_' + str(i), activation=activation)
            self.mlps.append(mlp)

    def preprocess(self,
                   data,
                   training=True,
                   vel_corr=None,
                   tape=None,
                   **kwargs):
        #
        # advection step
        #
        _pos, _vel, acc, feats, box, bfeats = data

        if vel_corr is not None:
            vel = tf.stop_gradient(vel_corr)
            pos = _pos + vel * self.timestep
        else:
            pos, vel = self.integrate_pos_vel(_pos, _vel, acc)

        #
        # preprocess features
        #
        fluid_feats = [tf.ones_like(pos[:, :1])]
        if self.use_vel:
            fluid_feats.append(vel)
        if self.use_acc:
            fluid_feats.append(acc)
        if self.use_feats:
            fluid_feats.append(feats)
        box_feats = [tf.ones_like(box[:, :1])]
        if self.use_box_feats:
            box_feats.append(bfeats)

        all_pos = tf.concat([pos, box], axis=0)
        if self.dens_feats or self.pres_feats:
            if self.dens_feats:
                dens = compute_density(all_pos, all_pos, self.query_radii, win=get_window_func("poly6"))
                fluid_feats.append(tf.expand_dims(dens[:tf.shape(pos)[0]], -1))
                box_feats.append(tf.expand_dims(dens[tf.shape(pos)[0]:], -1))
            if self.pres_feats:
                pres = compute_pressure(all_pos,
                                        all_pos,
                                        dens,
                                        self.rest_dens,
                                        win=get_window_func("poly6"),
                                        stiffness=self.stiffness)
                fluid_feats.append(tf.expand_dims(pres[:tf.shape(pos)[0]], -1))
                box_feats.append(tf.expand_dims(pres[tf.shape(pos)[0]:], -1))

        fluid_feats = tf.concat(fluid_feats, axis=-1)
        box_feats = tf.concat(box_feats, axis=-1)

        self.inp_feats = fluid_feats
        self.inp_bfeats = box_feats
        if tape is not None:
            tape.watch(self.inp_feats)
            tape.watch(self.inp_bfeats)

        return [pos, fluid_feats, box, box_feats]

    def forward(self, prev, data, training=True, **kwargs):
        pos, fluid_feats, box, box_feats = prev
        # compute the extent of the filters (the diameter)
        filter_extent = self.query_radii
        fluid_nns = self.radius_search(pos, pos, filter_extent)
        solid_nns = self.radius_search(box, pos, filter_extent)
        fluid_feats = self.fluid_encoder(xyz=pos, feats=fluid_feats, nns=fluid_nns)
        solid_feats = self.solid_encoder(xyz=box, feats=box_feats, nns=solid_nns, query=pos)
        feats = tf.nn.relu(fluid_feats + solid_feats)
        for i, mlp in enumerate(self.mlps):
            feats = mlp(feats)
        self.fluid_nns, self.solid_nns = fluid_nns, solid_nns
        return feats

    def postprocess(self, prev, data, training=True, vel_corr=None, **kwargs):
        #
        # postprocess output of network
        #
        pos, vel, acc, feats, box, bfeats = data
        out = prev
        #
        # scale to better match the scale of the output distribution
        #
        self.pos_correction = self.out_scale * out
        # self.obs = self.out_scale * out

        #
        # correct position and velocity
        #
        if vel_corr is not None:
            vel2 = tf.stop_gradient(vel_corr)
            pos2 = pos + vel2 * self.timestep
        else:
            pos2, vel2 = self.integrate_pos_vel(pos, vel, acc)

        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, self.pos_correction)

        try:
            if tf.reduce_sum(tf.abs(self.box - box)) < 0.01:
                pass
        except:
            self.calculate_boundary_mass(box)
            self.tree = cKDTree(self.box)

        box_masses = self.box_masses
        fluid_masses = tf.ones_like(pos[:, 0]) * self.fluid_mass
        densities = compute_density_with_box(pos2_corrected, fluid_masses, box, box_masses, self.query_radii)
        vel3_corrected = self.compute_XSPH_viscosity(self.fluid_nns, vel2_corrected, fluid_masses, densities,
                                                     self.viscosity, self.query_radii)
        vel3_corrected = self.compute_vorticity_confinement(self.fluid_nns, vel3_corrected, pos2_corrected,
                                                            self.query_radii)
        pos3_corrected, vel3_corrected = self.boundary_correction(pos2_corrected, vel3_corrected, self.tree, box,
                                                                  bfeats)
        self.pred_dens = densities
        return [pos3_corrected, vel3_corrected]

    def loss(self, results, data):
        loss = {}

        pred = results[0]
        target = data[1]

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        num_fluid_neighbors = tf.cast(
            self.fluid_nns.neighbors_row_splits[1:] - self.fluid_nns.neighbors_row_splits[:-1], tf.float32)
        num_solid_neighbors = tf.cast(
            self.solid_nns.neighbors_row_splits[1:] - self.solid_nns.neighbors_row_splits[:-1], tf.float32)

        for n, l in self.loss_fn.items():
            loss[n] = l(target,
                        pred,
                        pred_dens=self.pred_dens,
                        density0=self.rest_dens,
                        pre_steps=data[3],
                        num_fluid_neighbors=num_fluid_neighbors,
                        num_solid_neighbors=num_solid_neighbors,
                        input=data[0],
                        target_prev=data[2],
                        pos_correction=self.pos_correction)
        return loss

    def calculate_boundary_mass(self, box):
        dens = compute_density(box, radius=self.query_radii, win=get_window_func("poly6"))
        self.box_masses = self.rest_dens / dens
        self.box = box

    def loss_keys(self):
        return self.loss_fn.keys()
