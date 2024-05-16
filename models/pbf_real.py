import tensorflow as tf

import open3d.ml.tf as o3dml
import numpy as np
from abc import abstractmethod
from utils.tools.losses import get_loss, compute_density, compute_pressure, get_window_func, get_dilated_pos, \
    compute_transformed_dx
from utils.convolutions import ContinuousConv, PointSampling
from utils.tools.losses import get_window_func
from utils.tools.neighbor import combine_nns, reduce_subarrays_sum_multi

from .base_model import BaseModel


class PBFReal(BaseModel):
    def __init__(self,
                 name="PBFReal",
                 particle_radii=[0.025],
                 grav=-9.81,
                 transformation={},
                 loss={
                     "weighted_mse": {
                         "typ": "weighted_mse",
                         "fac": 1.0,
                         "gamma": 0.25,
                         "neighbor_scale": 0.025
                     }
                 },
                 timestep=0.0025,
                 query_radii=None,
                 density0=1000,
                 solver_iterations=10,
                 viscosity=0.02,
                 vorticity_fac=0.00001,
                 window_dens='cubic',
                 **kwargs):
        super().__init__(name=name,
                         timestep=timestep,
                         particle_radii=particle_radii,
                         transformation=transformation,
                         grav=grav,
                         window_dens=window_dens,
                         **kwargs)
        self.query_radii = particle_radii[0] * 4 if query_radii is None else query_radii
        self.m_neighborSearch = o3dml.layers.FixedRadiusSearch(ignore_query_point=True, return_distances=True)
        self.m_density0 = density0
        diameter = 2.0 * particle_radii[0]
        volume = diameter * diameter * diameter * 0.8
        self.fluid_mass = volume * self.m_density0
        self.m_maxIter = solver_iterations
        self.m_viscosity = viscosity
        self.vorticity_fac = vorticity_fac

        self.loss_fn = {}
        for l, v in loss.items():
            if v["typ"] == "dense":
                if not "radius" in v:
                    v["radius"] = self.query_radii
            self.loss_fn[l] = get_loss(**v)

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

        solid_masses = self.calculate_boundary_mass(box, self.query_radii, self.m_density0)

        if vel_corr is not None:
            vel = tf.stop_gradient(vel_corr)
            pos = _pos + vel * self.timestep
        else:
            pos, vel = self.integrate_pos_vel(_pos, _vel, acc)

        return [pos, vel, solid_masses]

    def forward(self, prev, data, training=True, **kwargs):
        pos, vel, solid_masses = prev
        _pos, _vel, acc, feats, box, bfeats = data

        fluid_nns = self.m_neighborSearch(pos, pos, self.query_radii)
        solid_nns = self.m_neighborSearch(box, pos, self.query_radii)

        group_neighbors = combine_nns(fluid_nns, solid_nns)
        group_masses = tf.concat([self.fluid_mass * tf.ones_like(pos[:, 0]), solid_masses], axis=0)

        iter = 0
        while True:
            group_position = tf.concat([pos, box], axis=0)
            # calculate density and lagrange multiplier
            density = compute_density(pos, group_position, self.query_radii, mass=group_masses,nns=group_neighbors)
            density_err = tf.clip_by_value(density - self.m_density0, 0, tf.float32.max)
            m_lambda = self.compute_lagrange_multiplier(group_position, group_masses, group_neighbors, density_err,
                                                        self.m_density0, self.query_radii)
            # self.m_lambda, self.m_density = m_lambda, m_density
            # perform density constraint
            m_deltaPos = self.solve_density_constraint(group_position, group_masses, group_neighbors, m_lambda,
                                                       self.m_density0, self.query_radii)
            # add the delta position to particles' position
            pos += m_deltaPos

            iter += 1
            if iter >= self.m_maxIter:
                break

        vel = (1.0 / self.timestep) * (pos - _pos)
        self.fluid_nns, self.solid_nns = fluid_nns, solid_nns
        self.densities = density

        return [pos, vel]

    def postprocess(self, prev, data, training=True, vel_corr=None, **kwargs):
        #
        # postprocess output of network
        #
        pos, vel = prev
        _pos, _vel, acc, feats, box, bfeats = data

        vel = self.compute_XSPH_viscosity(self.fluid_nns, pos, vel, self.fluid_mass, self.densities, self.m_viscosity,
                                          self.query_radii)
        # vel = self.compute_vorticity_confinement(self.fluid_nns, vel, pos, self.query_radii, self.vorticity_fac)

        return [pos, vel]

    def loss_keys(self):
        return self.loss_fn.keys()

    def loss(self, results, data):
        loss = {}

        pred = results[0]
        target = data[1]

        num_fluid_neighbors = tf.cast(
            self.fluid_nns.neighbors_row_splits[1:] - self.fluid_nns.neighbors_row_splits[:-1], tf.float32)
        num_solid_neighbors = tf.cast(
            self.solid_nns.neighbors_row_splits[1:] - self.solid_nns.neighbors_row_splits[:-1], tf.float32)

        for n, l in self.loss_fn.items():
            loss[n] = l(target,
                        pred,
                        num_fluid_neighbors=num_fluid_neighbors,
                        num_solid_neighbors=num_solid_neighbors,
                        input=data[0],
                        target_prev=data[2],
                        pre_steps=data[3])

        return loss
