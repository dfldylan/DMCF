import tensorflow as tf
from scipy.spatial import cKDTree
import open3d.ml.tf as o3dml
import numpy as np
from utils.tools.losses import get_window_func, compute_density, compute_pressure
from utils.tools.neighbor import reduce_subarrays_sum_multi

from .pbf_real import PBFReal


class PolarConv(tf.keras.layers.Layer):
    def __init__(self, filters, radius_search_ignore_query_points=True, activation=None, **kwargs):
        super(PolarConv, self).__init__(**kwargs)
        self.out_dims = filters
        self.activation = activation
        self.fixed_radius_search = o3dml.layers.FixedRadiusSearch(ignore_query_point=radius_search_ignore_query_points)

    def build(self, inp_features_shape):
        # 创建权重矩阵，这里假设权重是可学习的
        self.kernel = self.add_weight("kernel", shape=[4, inp_features_shape[-1], self.out_dims])

    def call(self, inp_features, inp_positions, out_positions, extents):
        # 这里我们只是简单地计算每个输出点的邻居
        # 您可能需要定义一个更复杂的函数来找到正确的邻居
        nns = self.fixed_radius_search(inp_positions, out_positions, extents)
        neighbors_index, neighbors_row_splits, _ = nns
        neighbors_index = tf.cast(neighbors_index, tf.int32)
        neighbors_row_splits = tf.cast(neighbors_row_splits, tf.int32)
        # 获取每个query点的邻居数
        neighbors_counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        # 对query进行扩展以匹配neighbors_index的形状
        expanded_query = tf.repeat(out_positions, neighbors_counts, axis=0)
        # 确保diff的维度和xyz对应
        diff = tf.gather(inp_positions, neighbors_index) - expanded_query

        # 这里我们只是简单地计算极坐标和权重
        # 您可能需要定义一个更复杂的函数来计算正确的极坐标和权重
        polar_coords = self.cartesian_to_polar(diff, extents)  # [N, 4]
        # weights = self.calculate_weights(polar_coords, self.kernel)  # [N, in, out]
        # 使用tf.gather收集输入特征
        # inp_features_gather = tf.gather(inp_features, neighbors_index)  # [N, in_dims]
        # 使用einsum一步完成乘法和求和操作
        weighted_features = tf.einsum('nx,xio,ni->no', polar_coords, self.kernel,
                                      tf.gather(inp_features, neighbors_index))  # [N, out]
        # 如果需要，再进行额外的reduce操作
        out_features = reduce_subarrays_sum_multi(weighted_features, neighbors_row_splits)

        if self.activation:
            out_features = self.activation(out_features)
        return out_features, nns

    def cartesian_to_polar(self, cartesian_coords, extents):
        # 这是一个简单的从笛卡尔坐标到极坐标的转换
        # 您可能需要根据实际情况修改这个函数
        r = tf.norm(cartesian_coords, axis=-1)
        cartesian_coords_normalized = cartesian_coords / r[..., tf.newaxis]
        sin_theta = cartesian_coords_normalized[..., 0]
        cos_theta = cartesian_coords_normalized[..., 2]
        cos_phi = cartesian_coords_normalized[..., 1]
        # Weighted feature
        r_normalized = r / extents  # 防止除以零
        return tf.stack([r_normalized, sin_theta, cos_theta, cos_phi], axis=-1)

    def calculate_weights(self, polar_coords, kernel):
        return tf.nn.softmax(tf.einsum('nx,xio->nio', polar_coords, kernel), axis=1)


class PolarNetG(PBFReal):
    def __init__(self,
                 name="PolarNetG",
                 timestep=0.02,
                 grav=-9.81,
                 rest_dens=1000.0,
                 viscosity=0.02,
                 particle_radii=[0.025],
                 query_radii=None,
                 use_mass=True,
                 use_acc=False,
                 use_vel=True,
                 use_feats=False,
                 use_box_feats=True,
                 transformation={},
                 ignore_query_points=True,
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
                 channels=16,
                 layer_channels=[48, 64, 64, 3],
                 out_scale=[0.01, 0.01, 0.01],
                 window_dens='cubic',
                 **kwargs):

        super().__init__(name=name,
                         timestep=timestep,
                         particle_radii=particle_radii,
                         transformation=transformation,
                         grav=grav,
                         loss=loss,
                         query_radii=query_radii,
                         density0=rest_dens,
                         viscosity=viscosity,
                         window_dens=window_dens,
                         **kwargs)
        # self.query_radii = particle_radii[0] * 2 if query_radii is None else query_radii
        diameter = 2.0 * particle_radii[0]
        volume = diameter ** 3
        self.fluid_mass = volume * self.m_density0

        self.use_mass = use_mass
        self.use_vel = use_vel
        self.use_acc = use_acc
        self.use_feats = use_feats
        self.use_box_feats = use_box_feats
        self.dens_feats = dens_feats
        self.pres_feats = pres_feats
        self.stiffness = stiffness
        self.out_scale = tf.constant(out_scale)
        self.channels = channels
        self.ignore_query_points = ignore_query_points
        self.layer_channels = layer_channels
        self.viscosity = viscosity

        self.radius_search = o3dml.layers.FixedRadiusSearch(ignore_query_point=True)

        self._all_convs = []

        self.fluid_convs = self.get_cconv(name='fluid_obs',
                                          filters=channels,
                                          activation=None)

        self.fluid_dense = tf.keras.layers.Dense(units=channels,
                                                 name="fluid_dense",
                                                 activation=None)

        self.obs_convs = self.get_cconv(name='obs_conv',
                                        filters=channels,
                                        activation=None)

        self.obs_dense = tf.keras.layers.Dense(units=channels,
                                               name="obs_dense",
                                               activation=None)

        self.layers_ops = [None]
        for i in range(1, len(self.layer_channels)):
            layer_ops = dict()
            ch = self.layer_channels[i]
            if i != 1 and i != len(self.layer_channels) - 1:
                ch = int(self.layer_channels[i] / 2)
            if i != len(self.layer_channels) - 1:
                conv = self.get_cconv(name='conv{0}'.format(i),
                                      filters=ch,
                                      activation=None,
                                      ignore_query_points=self.ignore_query_points)
                layer_ops['conv'] = conv
            if i != 1:
                dense = tf.keras.layers.Dense(units=ch,
                                              name="dense{0}".format(i),
                                              activation=None)
                layer_ops['dense'] = dense
            self.layers_ops.append(layer_ops)

    def get_cconv(self,
                  name,
                  activation=None,
                  ignore_query_points=None,
                  **kwargs):

        if ignore_query_points is None:
            ignore_query_points = self.ignore_query_points
        conv = PolarConv(
            name=name,
            activation=activation,
            radius_search_ignore_query_points=ignore_query_points,
            **kwargs)

        self._all_convs.append((name, conv))
        return conv

    def preprocess(self,
                   data,
                   training=True,
                   vel_corr=None,
                   tape=None,
                   **kwargs):
        pos, vel, solid_masses = super(PolarNetG, self).preprocess(data, training, vel_corr, tape, **kwargs)
        _pos, _vel, acc, feats, box, bfeats = data
        self.solid_masses = 1.2 * solid_masses
        #
        # preprocess features
        #
        # compute the extent of the filters (the diameter)
        fluid_feats = [tf.ones_like(pos[:, :1])]
        box_feats = [tf.ones_like(box[:, :1])]
        if self.use_mass:
            fluid_feats.append(fluid_feats[0] * self.fluid_mass)
            box_feats.append(self.solid_masses[:, tf.newaxis])
        if self.use_vel:
            fluid_feats.append(vel)
        if self.use_acc:
            fluid_feats.append(acc)
        if self.use_feats:
            fluid_feats.append(feats)
        if self.use_box_feats:
            box_feats.append(bfeats)

        all_pos = tf.concat([pos, box], axis=0)
        self.all_pos = all_pos
        if self.dens_feats or self.pres_feats:
            dens = compute_density(all_pos, all_pos, self.query_radii, win=get_window_func(self.window_dens))
            if self.dens_feats:
                fluid_feats.append(tf.expand_dims(dens[:tf.shape(pos)[0]], -1))
                box_feats.append(tf.expand_dims(dens[tf.shape(pos)[0]:], -1))
            if self.pres_feats:
                pres = compute_pressure(all_pos,
                                        all_pos,
                                        dens,
                                        self.m_density0,
                                        win=get_window_func(self.window_dens),
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

        ans_conv, f_nns = self.fluid_convs(fluid_feats, pos, all_pos, self.query_radii)
        ans_dense = self.fluid_dense(fluid_feats)

        ans_obs, s_nns = self.obs_convs(box_feats, box, all_pos, self.query_radii)
        ans_dense_obs = self.obs_dense(box_feats)

        ans_dense = tf.concat([ans_dense, ans_dense_obs], axis=0)

        feats = tf.concat([ans_conv, ans_obs, ans_dense], axis=-1)

        self.fluid_nns, self.solid_nns = f_nns, s_nns
        return [pos, vel, feats]

    def forward(self, prev, data, training=True, **kwargs):
        pos, vel, feats = prev
        _pos, _vel, acc, _feats, box, bfeats = data
        # feats = feats[:tf.shape(pos)[0]]

        ans_convs = [feats]  # [channels*3]
        for i in range(1, len(self.layers_ops)):
            conv = self.layers_ops[i].get('conv', None)
            dense = self.layers_ops[i].get('dense', None)
            feats = tf.keras.activations.relu(ans_convs[-1])
            ans = []
            if conv is not None:
                if i == 1:
                    ans_conv, _ = conv(feats, self.all_pos, pos, self.query_radii)
                else:
                    ans_conv, self.fluid_nns = conv(feats, pos, pos, self.query_radii)
                ans.append(ans_conv)
            if dense is not None:
                ans_dense = dense(feats)
                ans.append(ans_dense)
            ans = tf.concat(ans, axis=-1)
            ans_convs.append(ans)

        out = ans_convs[-1]
        if out.shape[-1] == 1:
            out = tf.repeat(out, 3, axis=-1)
        elif out.shape[-1] == 2:
            out = tf.concat([out, out[:, :1]], axis=-1)

        #
        # scale to better match the scale of the output distribution
        #
        pcnt = tf.shape(pos)[0]
        self.pos_correction = self.out_scale * out[:pcnt]
        self.obs = self.out_scale * out[pcnt:]

        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(_pos, _vel, pos, vel, self.pos_correction)

        return [pos2_corrected, vel2_corrected]

    def postprocess(self, prev, data, training=True, vel_corr=None, **kwargs):
        #
        # postprocess output of network
        #
        pos, vel = prev
        _pos, _vel, acc, feats, box, bfeats = data

        group_position = tf.concat([pos, box], axis=0)
        group_neighbors = self.radius_search(group_position, pos, self.query_radii)
        group_masses = tf.concat([self.fluid_mass * tf.ones_like(pos[:, 0]), self.solid_masses], axis=0)
        self.densities, _ = self.compute_density_with_mass(group_position, group_masses, group_neighbors,
                                                           self.m_density0, self.query_radii)

        # pos, vel = super(PolarNetG, self).postprocess(prev, data, training, vel_corr, **kwargs)
        return [pos, vel]

    def loss(self, results, data):
        loss = {}

        pos, vel = results
        target, target_vel = data[1], data[4]

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        fluid_num = tf.shape(pos)[0]
        num_fluid_neighbors = tf.cast(
            self.fluid_nns.neighbors_row_splits[1:] - self.fluid_nns.neighbors_row_splits[:-1], tf.float32)[:fluid_num]
        num_solid_neighbors = tf.cast(
            self.solid_nns.neighbors_row_splits[1:] - self.solid_nns.neighbors_row_splits[:-1], tf.float32)[:fluid_num]

        num_fluid_neighbors, num_solid_neighbors = \
            tf.stop_gradient(num_fluid_neighbors), tf.stop_gradient(num_solid_neighbors)

        target = tf.concat([target, target_vel], axis=-1)
        pred = tf.concat([pos, vel], axis=-1)
        for n, l in self.loss_fn.items():
            loss[n] = l(target,
                        pred,
                        pred_dens=self.densities,
                        density0=self.m_density0,
                        pre_steps=data[3],
                        num_fluid_neighbors=num_fluid_neighbors,
                        num_solid_neighbors=num_solid_neighbors,
                        input=data[0],
                        target_prev=data[2],
                        pos_correction=self.pos_correction)
        return loss

    def loss_keys(self):
        return self.loss_fn.keys()
