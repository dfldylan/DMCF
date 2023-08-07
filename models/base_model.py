import numpy as np
import yaml
import tensorflow as tf
from os.path import join, exists, dirname, abspath
from abc import ABC, abstractmethod

from o3d.utils import Config
from utils.tools.losses import get_window_func, compute_density
from utils.tools.neighbor import neighbors_mask, reduce_subarrays_sum_multi


@tf.function
def align_vector(v0, v1):
    v0_norm = v0 / (tf.norm(v0) + 1e-9)
    v1_norm = v1 / (tf.norm(v1) + 1e-9)

    v = tf.linalg.cross(v0_norm, v1_norm)
    c = tf.tensordot(v0_norm, v1_norm, 1)
    s = tf.norm(v)

    if s < 1e-6:
        return tf.eye(3) * (-1.0 if c < 0 else 1.0)

    vx = tf.convert_to_tensor([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]],
                               [-v[1], v[0], 0.0]])

    r = tf.eye(3) + vx + tf.tensordot(vx, vx, 1) / (1 + c)
    return r


class BaseModel(ABC, tf.keras.Model):
    """Base class for models.

    All models must inherit from this class and implement all functions to be
    used with a pipeline.

    Args:
        **kwargs: Configuration of the model as keyword arguments.
    """

    def __init__(self,
                 name,
                 timestep,
                 particle_radii,
                 transformation,
                 grav,
                 window_dens=None,
                 **kwargs):
        super().__init__(name=name)
        # physics setup
        self.grav = grav
        self.timestep = timestep
        self.particle_radii = particle_radii
        self.transformation = transformation
        self.window_dens = window_dens

        self.cfg = Config(kwargs)

    def call(self, data, training=True, **kwargs):
        d = self.transform(data, training=training, **kwargs)
        x = self.preprocess(d, training=training, **kwargs)
        x = self.forward(x, d, training=training, **kwargs)
        x = self.postprocess(x, d, training=training, **kwargs)
        x = self.inv_transform(x, data, training=training, **kwargs)
        return x

    @abstractmethod
    def forward(self, prev, data, training=True, **kwargs):
        return

    @abstractmethod
    def loss(self, results, data):
        """Computes the loss given the network input and outputs.

        Args:
            results: This is the output of the model.
            inputs: This is the input to the model.

        Returns:
            Returns the loss value.
        """
        return {}

    def get_optimizer(self, cfg):
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            cfg['lr_boundaries'], cfg['lr_values'])

        optimizer = tf.optimizers.Adam(learning_rate=learning_rate_fn,
                                       epsilon=1e-6)
        return optimizer

    def transform(self, data, training=True, **kwargs):
        pos, vel, acc, feats, box, bfeats = data

        if "translate" in self.transformation:
            translate = tf.constant(self.transformation["translate"],
                                    tf.float32)
            pos += translate
            box += translate

        if "scale" in self.transformation:
            scale = tf.constant(self.transformation["scale"], tf.float32)
            pos *= scale
            box *= scale
            vel *= scale
            if acc is not None:
                acc *= scale

        if "grav_eqvar" in self.transformation:
            grav_eqvar = tf.constant(self.transformation["grav_eqvar"],
                                     tf.float32)
            # WARNING: assuming same gravity for all particles for one sequence
            self.R = align_vector(grav_eqvar, acc[0])
            pos = tf.linalg.matmul(pos, self.R)
            vel = tf.linalg.matmul(vel, self.R)
            acc = tf.linalg.matmul(acc, self.R)
            box = tf.linalg.matmul(box, self.R)
            bfeats = tf.linalg.matmul(bfeats, self.R)

        return [pos, vel, acc, feats, box, bfeats]

    def inv_transform(self, prev, data, **kwargs):
        pos, vel = prev

        if "grav_eqvar" in self.transformation:
            # WARNING: assuming same gravity for all particles for one sequence
            R = tf.transpose(self.R)
            pos = tf.linalg.matmul(pos, R)
            vel = tf.linalg.matmul(vel, R)

        if "scale" in self.transformation:
            scale = tf.constant(self.transformation["scale"], tf.float32)
            pos /= tf.maximum(scale, 1e-5)
            vel /= tf.maximum(scale, 1e-5)

        if "translate" in self.transformation:
            translate = tf.constant(self.transformation["translate"],
                                    tf.float32)
            pos -= translate

        return pos, vel

    def preprocess(self, data, training=True, **kwargs):
        """Preprocessing step.

        Args:
            input: Input of model

        Returns:
            Returns modified input.
        """

        return input

    def postprocess(self, prev, data, training=True, **kwargs):
        """Preprocessing step.

        Args:
            input: Output of model

        Returns:
            Returns modified output.
        """

        return input

    def integrate_pos_vel(self, pos1, vel1, acc1=None):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * (acc1 if acc1 is not None else tf.constant(
            [0, self.grav, 0]))
        pos2 = pos1 + dt * vel2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_XSPH_viscosity(self, fluid_nns, pos, velocities, fluid_mass, densities, viscosity, radius,
                               win=get_window_func("poly6")):
        neighbors_index, neighbors_row_splits, _ = fluid_nns
        # Get neighbor positions, velocities, masses, and densities
        neighbors_velocities = tf.gather(velocities, neighbors_index)
        # neighbors_masses = tf.gather(masses, neighbors_index)
        neighbors_densities = tf.gather(densities, neighbors_index)
        # Compute tmp values for all particles and neighbors
        neighbors_counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        tmp = tf.repeat(velocities, neighbors_counts, axis=0) - neighbors_velocities
        dist = tf.repeat(pos, neighbors_counts, axis=0) - tf.gather(pos, neighbors_index)
        dist = tf.reduce_sum(tf.square(dist), axis=-1) / radius ** 2
        tmp *= tf.expand_dims(win(dist) * fluid_mass / neighbors_densities, axis=-1)
        # Compute sum_value for all particles
        delta_velocity = -reduce_subarrays_sum_multi(tmp, neighbors_row_splits) * viscosity
        return velocities + delta_velocity

    def _gradient(self, vec, radius, win=get_window_func("cubic_grad")):
        vec_squared_sum = tf.reduce_sum(vec ** 2, axis=-1, keepdims=True)
        gradients = vec / (tf.sqrt(vec_squared_sum) * radius) * win(vec_squared_sum / radius ** 2)
        return gradients

    def compute_vorticity_confinement(self, fluid_nns, velocities, positions, radius, fac=0.00001,
                                      win=get_window_func("cubic_grad")):

        timeStep = self.timestep
        neighbors_index, neighbors_row_splits, _ = fluid_nns
        neighbors_counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        # 对于每个粒子，计算与其邻居的位置和速度差
        velGap = tf.gather(velocities, neighbors_index) - tf.repeat(velocities, neighbors_counts, axis=0)
        posGap = tf.repeat(positions, neighbors_counts, axis=0) - tf.gather(positions, neighbors_index)
        # 计算渐变和交叉积
        gradients = self._gradient(posGap, radius, win)
        curl = reduce_subarrays_sum_multi(tf.linalg.cross(velGap, gradients), neighbors_row_splits)
        # 分别向x,y,z方向偏移，再计算渐变和交叉积
        posGap_shifted_x = posGap + [0.01, 0, 0]
        posGap_shifted_y = posGap + [0, 0.01, 0]
        posGap_shifted_z = posGap + [0, 0, 0.01]
        gradients_shifted_x = self._gradient(posGap_shifted_x, radius, win)
        gradients_shifted_y = self._gradient(posGap_shifted_y, radius, win)
        gradients_shifted_z = self._gradient(posGap_shifted_z, radius, win)
        curl_shifted_x = reduce_subarrays_sum_multi(tf.linalg.cross(velGap, gradients_shifted_x), neighbors_row_splits)
        curl_shifted_y = reduce_subarrays_sum_multi(tf.linalg.cross(velGap, gradients_shifted_y), neighbors_row_splits)
        curl_shifted_z = reduce_subarrays_sum_multi(tf.linalg.cross(velGap, gradients_shifted_z), neighbors_row_splits)
        # 计算N和力
        curl_len = tf.norm(curl, axis=-1, keepdims=True)
        N = [tf.norm(curl_shifted_x, axis=-1, keepdims=True) - curl_len,
             tf.norm(curl_shifted_y, axis=-1, keepdims=True) - curl_len,
             tf.norm(curl_shifted_z, axis=-1, keepdims=True) - curl_len]
        N = tf.concat(N, axis=-1)
        N_norm = tf.norm(N, axis=-1, keepdims=True)
        # 创建一个掩码，用于识别非零范数
        mask = tf.squeeze(N_norm != 0, axis=-1)
        # 使用掩码来选择元素
        N_filtered = tf.boolean_mask(N, mask)
        N_norm_filtered = tf.boolean_mask(N_norm, mask)
        curl_filtered = tf.boolean_mask(curl, mask)
        # 对滤波后的N执行归一化
        N_normalized = N_filtered / N_norm_filtered
        # 计算力和速度变化
        force = fac * tf.linalg.cross(N_normalized, curl_filtered)
        delta_velocity = timeStep * force
        # 获取满足掩码条件的索引
        indices = tf.where(mask)
        # 将delta_velocity添加到原始速度的对应位置
        final_velocities = tf.tensor_scatter_nd_add(velocities, indices, delta_velocity)
        return final_velocities

    def boundary_correction(self, pos, vel, tree, box, box_normals):
        restitutionCoefficient = 0.25
        _frictionCoeffient = 0.04
        # Get nearest box particles and normal vectors for each fluid particle
        distances, indices = tree.query(pos, k=1)
        indices = tf.convert_to_tensor(indices, dtype=tf.int32)
        box_nearest_normal = tf.gather(box_normals, indices)
        box_nearest = tf.gather(box, indices)
        dist_rel = pos - box_nearest
        dot_normal_pos = tf.reduce_sum(box_nearest_normal * dist_rel, axis=1, keepdims=True)
        mask = dot_normal_pos[:, 0] < 0
        # If a fluid particle penetrated the boundary, calculate a target position for it on the boundary
        pos = tf.where(mask[:, tf.newaxis], box_nearest + box_nearest_normal * 0.025, pos)
        # Calculate the dot product of the fluid particle's relative velocity and the boundary particle's normal vector
        dot_normal_vel = tf.reduce_sum(box_nearest_normal * vel, axis=1, keepdims=True)
        mask2 = tf.where(tf.logical_and(mask, dot_normal_vel[:, 0] < 0))
        # Compute relative velocity, decompose it into normal and tangential components
        mask2_vel = tf.gather(vel, mask2)
        mask2_result2 = tf.gather(dot_normal_vel, mask2)
        mask2_box_nearest_normal = tf.gather(box_nearest_normal, mask2)
        relativeVelN = mask2_result2 * mask2_box_nearest_normal
        relativeVelT = mask2_vel - relativeVelN
        # Apply a restitution coefficient to the normal component of the relative velocity and calculate friction for the tangential component
        relativeVelN *= -restitutionCoefficient
        frictionScale = tf.clip_by_value(
            1.0 - _frictionCoeffient * tf.norm(relativeVelN) / tf.norm(relativeVelT + 1e-5), 0.0, 1.0)
        relativeVelT *= frictionScale
        newVelocity = relativeVelN + relativeVelT
        # Correct fluid particle's position and velocity
        vel = tf.tensor_scatter_nd_update(vel, mask2, newVelocity[:, 0])
        return pos, vel

    def calculate_boundary_mass(self, box, query_radii, rest_dens, win=get_window_func("cubic")):
        dens = compute_density(box, radius=query_radii, win=win)
        box_masses = rest_dens / dens
        return box_masses

    def compute_density_with_mass(self, mass, neighbors, density0, radius, win=get_window_func("cubic")):
        neighbors_index, neighbors_row_splits, neighbors_distance = neighbors
        density = tf.gather(mass, neighbors_index) * win(neighbors_distance / radius ** 2)
        density = reduce_subarrays_sum_multi(density, neighbors_row_splits)
        density += mass[:tf.shape(density)[0]] * win(0.0)
        density_err = tf.clip_by_value(density - density0, 0, tf.float32.max)
        return density, density_err

    def compute_lagrange_multiplier(self, position, mass, neighbors, density_err, density0, radius, eps=1.0e-6):
        constraint = density_err / density0
        lambda_val = tf.zeros_like(constraint)
        mask = tf.math.abs(constraint) > eps
        neighbors = neighbors_mask(neighbors, mask)
        neighbors_index, neighbors_row_splits, _ = neighbors
        neighbors_counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        neighbor_positions = tf.gather(position, neighbors_index)
        position_diffs = tf.repeat(tf.boolean_mask(position[:tf.shape(constraint)[0]], mask), neighbors_counts,
                                   axis=0) - neighbor_positions
        neighbor_masses = tf.gather(mass, neighbors_index)
        grad_cj = neighbor_masses[:, None] / density0 * self._gradient(position_diffs, radius)
        grad_ci = reduce_subarrays_sum_multi(grad_cj, neighbors_row_splits)
        sum_grad_cj = reduce_subarrays_sum_multi(tf.reduce_sum(tf.square(grad_cj), axis=-1),
                                                 neighbors_row_splits) + tf.reduce_sum(tf.square(grad_ci), axis=-1)
        lambda_val_masked = -tf.boolean_mask(constraint, mask) / (sum_grad_cj + eps)
        lambda_val = tf.tensor_scatter_nd_update(lambda_val, tf.where(mask), lambda_val_masked)
        return lambda_val

    def solve_density_constraint(self, position, mass, neighbors, lambda_vals, density0, radius):
        neighbors_index, neighbors_row_splits, _ = neighbors
        # Compute grad_cj for each neighbor.
        neighbors_counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
        neighbor_positions = tf.gather(position, neighbors_index)
        position_repeat = tf.repeat(position[:tf.shape(lambda_vals)[0]], neighbors_counts, axis=0)
        position_diffs = position_repeat - neighbor_positions
        # shape: [numFluidParticle, num_neighbors, 3]
        neighbor_masses = tf.gather(mass, neighbors_index)
        grad_cj = neighbor_masses[:, None] / density0 * self._gradient(position_diffs, radius)
        # shape: [numFluidParticle, num_neighbors, 3]
        # Compute lambda*grad_cj for each neighbor.
        lambda_i = tf.repeat(lambda_vals, neighbors_counts, axis=0)  # shape: [numFluidParticle]
        lambda_j = tf.gather(tf.pad(lambda_vals, paddings=[[0, tf.shape(position)[0] - tf.shape(lambda_vals)[0]]]),
                             neighbors_index)
        # shape: [total_num_neighbors]
        # shape: [total_num_neighbors]
        lambda_ij = lambda_i + lambda_j
        lambda_grad_cj = lambda_ij[:, None] * grad_cj  # shape: [numFluidParticle, num_neighbors, 3]
        # Sum up lambda*grad_cj for each particle.
        deltaPos = reduce_subarrays_sum_multi(lambda_grad_cj, neighbors_row_splits)  # shape: [numFluidParticle, 3]
        return deltaPos
