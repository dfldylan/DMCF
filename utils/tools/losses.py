import open3d.ml.tf as ml3d
import numpy as np
import tensorflow as tf
from utils.tools.sampling import gather_point, farthest_point_sample
from utils.tools.tf_approxmatch import approx_match, match_cost
from functools import partial


def get_window_func(typ, fac=1.0, **kwargs):
    if typ == "poly6":
        def func(q):
            return fac * tf.clip_by_value((1 - q) ** 3, 0, 1)

    elif typ == "cubic":  # fac = 6.0 / (np.pi * h**3)
        def func(q):
            q_sqrt = tf.sqrt(q)
            return fac * 4 / 3 * tf.where(q <= 1,
                                          tf.where(q_sqrt <= 0.5,
                                                   6 * (q_sqrt ** 3 - q) + 1,
                                                   2 * (1 - q_sqrt) ** 3),
                                          tf.zeros_like(q_sqrt))

    elif typ == "linear":
        def func(q):
            q_sqrt = tf.sqrt(q)
            return fac * (1 - q_sqrt)

    elif typ == "peak":
        def func(q):
            q_sqrt = tf.sqrt(q)
            return fac * (1 - 2 * q_sqrt + q)

    elif typ == "cubic_grad":  # fac = 6.0 / (np.pi * h**3) (r/(rl*h))
        def func(q):
            q_sqrt = tf.sqrt(q)
            return fac * 4 / 3 * tf.where(q <= 1,
                                          tf.where(q_sqrt <= 0.5,
                                                   18 * q - 12 * q_sqrt,
                                                   -6 * (1 - q_sqrt) ** 2),
                                          tf.zeros_like(q_sqrt))

    elif typ is None:
        func = None

    else:
        raise NotImplementedError(f"Window function type '{typ}' is not implemented.")

    return func


def get_loss(typ, fac=1.0, **kwargs):
    if typ == "mse":

        def f(target, pred, **kw):
            pre_f = tf.exp(-kwargs.get("pre_scale", 0.0) *
                           tf.cast(kw.get("pre_steps"), tf.float32))
            diff = (tf.reduce_sum(
                (target - pred) ** 2, axis=-1) + 1e-9) ** kwargs.get("gamma", 0.5)
            return fac * tf.reduce_mean(pre_f * diff)

        return f
    elif typ == "weighted_mse":

        def f(target, pred, **kw):
            pre_f = tf.exp(-kwargs.get("pre_scale", 0.0) * tf.cast(kw.get("pre_steps"), tf.float32))
            fluid_neighbor_scale = kwargs.get("neighbor_scale", 1.0)
            solid_neighbor_scale = kwargs.get("box_neighbor_scale", 1.0)
            fluid_method = kwargs.get("fluid_method", 'exp')
            x = kw.get("num_fluid_neighbors", 0.0)
            if fluid_method == 'exp':
                y = 1 + tf.exp(-x)
            elif fluid_method == 'reciprocal':
                y = 1 + (x + 1) ** (-1)
            else:
                raise NotImplementedError(f"Window function type '{fluid_method}' is not implemented.")
            fluid_importance = fluid_neighbor_scale * y

            solid_method = kwargs.get("solid_method", 'linear')
            x = kw.get("num_solid_neighbors", 0.0)
            if solid_method == 'exp':
                y = tf.exp(x)
            elif solid_method == 'linear':
                y = 1 + x
            elif solid_method == 'square':
                y = 1 + x ** 2
            else:
                raise NotImplementedError(f"Window function type '{solid_method}' is not implemented.")
            solid_importance = solid_neighbor_scale * y

            scale = -kwargs.get("scale", 1.0)
            diff = (tf.reduce_sum(((target - pred) * scale) ** 2, axis=-1) + 1e-9) ** kwargs.get("gamma", 0.5)
            return fac * tf.reduce_mean(pre_f * fluid_importance * solid_importance * diff)

        return f
    elif typ == "dense":
        win = get_window_func(kwargs.pop("win", None))
        return partial(density_loss, win=win, **kwargs)
    elif typ == "vel":

        def f(target, pred, **kw):
            inp = kw.get("input")[0]
            prev = kw.get("target_prev")
            diff = (tf.reduce_sum(
                ((target - prev) - (pred - inp)) ** 2, axis=-1) +
                    1e-9) ** kwargs.get("gamma", 0.5)
            return fac * tf.reduce_mean(diff)

        return f
    elif typ == "weighted_vel":

        def f(target, pred, **kw):
            inp = kw.get("input")[0]
            prev = kw.get("target_prev")
            importance = tf.exp(-kwargs.get("neighbor_scale", 1.0) *
                                kw.get("num_fluid_neighbors"))
            diff = (tf.reduce_sum(
                ((target - prev) - (pred - inp)) ** 2, axis=-1) +
                    1e-9) ** kwargs.get("gamma", 0.5)
            return fac * tf.reduce_mean(importance * diff)

        return f
    elif typ == "momentum":

        def f(target, pred, **kw):
            return fac * tf.reduce_mean(kw.get("pos_correction"))

        return f
    elif typ == "chamfer":
        return partial(chamfer_distance, **kargs)
    elif typ == "emd":
        return partial(emd_loss, **kargs)
    elif typ == "hist":
        return compare_dist
    elif typ == "dense_pbf":
        return partial(density_loss_pbf, fac=fac, **kwargs)
    else:
        raise NotImplementedError()


def _grid_pos(pos, res):
    zs, ys, xs = res[0], res[1], res[2]
    zr, yr, xr = tf.meshgrid(tf.range(zs),
                             tf.range(ys),
                             tf.range(xs),
                             indexing='ij')

    grid_pos = tf.cast(tf.stack([xr, yr, zr], axis=-1), tf.float32) + 0.5
    grid_pos = grid_pos / tf.cast(res[::-1], tf.float32)
    grid_pos = tf.reshape(grid_pos, (-1, 3))

    grid_pos = grid_pos * tf.cast(res != 1, tf.float32)

    pos = pos[tf.reduce_all((pos >= 0.0) & (pos < 1.0), axis=-1)]
    mask = tf.scatter_nd(tf.cast(pos[:, ::-1] * tf.cast(res[::-1], tf.float32),
                                 tf.int32),
                         tf.ones((tf.shape(pos)[0], 1), tf.bool),
                         shape=[xs, ys, zs, 1])
    mask = tf.reshape(mask, (-1,))

    return grid_pos[mask]


def grid_pos(pos, voxel_size, centralize=False, pad=0, hyst=0.1):
    if centralize:
        center = tf.reduce_mean(pos, axis=0)
        pos = pos - center

    # discretize
    dpos = tf.concat([
        tf.cast(tf.floor(pos / tf.maximum(voxel_size, 1e-5) -
                         tf.where(voxel_size >= 1e-5, hyst, tf.constant(0.0))),
                dtype=tf.int32),
        tf.cast(tf.floor(pos / tf.maximum(voxel_size, 1e-5) +
                         tf.where(voxel_size >= 1e-5, hyst, tf.constant(0.0))),
                dtype=tf.int32)
    ],
        axis=0)
    offset = tf.stack(tf.meshgrid(tf.where(voxel_size[0] >= 1e-5,
                                           tf.range(-pad, 2 + pad),
                                           tf.range(0, 1)),
                                  tf.where(voxel_size[1] >= 1e-5,
                                           tf.range(-pad, 2 + pad),
                                           tf.range(0, 1)),
                                  tf.where(voxel_size[2] >= 1e-5,
                                           tf.range(-pad, 2 + pad),
                                           tf.range(0, 1)),
                                  indexing='ij'),
                      axis=-1)
    offset = tf.reshape(offset, (1, -1, 3))
    dpos = tf.expand_dims(dpos, axis=1) + offset
    dpos = tf.reshape(dpos, (-1, 3))

    # remove duplicated points
    minp = tf.reduce_min(dpos, axis=0)
    maxp = tf.reduce_max(dpos, axis=0) - minp + 1
    idx = tf.reduce_sum((dpos - minp) * [1, maxp[0], maxp[0] * maxp[1]],
                        axis=-1)
    idx = tf.unique(idx)[0]
    gpos = tf.stack(
        [idx % maxp[0], idx // maxp[0] % maxp[1], idx // (maxp[0] * maxp[1])],
        axis=-1) + minp

    if centralize:
        gpos = tf.cast(gpos, tf.float32) * voxel_size + center
    else:
        gpos = tf.cast(gpos, tf.float32) * voxel_size + voxel_size / 2

    return gpos


def grid_pos_bnds(pos, voxel_size, centralize=False):
    if centralize:
        minpos = tf.reduce_min(pos, axis=0)
        maxpos = tf.maximum(tf.reduce_max(pos, axis=0) - minpos, 1e-7)
        r = tf.round(maxpos / tf.maximum(voxel_size, 1e-5))

        # discretize
        dpos = tf.cast(tf.round((pos - minpos) / maxpos * r), dtype=tf.int32)
    else:
        # discretize
        dpos = tf.cast(tf.round(pos / tf.maximum(voxel_size, 1e-5)),
                       dtype=tf.int32)

    # remove duplicated points
    minp = tf.reduce_min(dpos, axis=0)
    maxp = tf.reduce_max(dpos, axis=0) - minp + 1
    idx = tf.reduce_sum((dpos - minp) * [1, maxp[0], maxp[0] * maxp[1]],
                        axis=-1)
    idx = tf.unique(idx)[0]
    gpos = tf.stack(
        [idx % maxp[0], idx // maxp[0] % maxp[1], idx // (maxp[0] * maxp[1])],
        axis=-1) + minp

    if centralize:
        gpos = tf.cast(gpos, tf.float32) / tf.maximum(r,
                                                      1e-7) * maxpos + minpos
    else:
        gpos = tf.cast(gpos, tf.float32) * voxel_size + voxel_size / 2
    return gpos


def subsample(pos,
              stride,
              val=None,
              res=None,
              centralize=False,
              pad=0,
              hyst=0.1):
    vals = None
    if stride == 1:
        # no sub-sampling
        pcnt = tf.shape(pos)[0]
        dilated_pos = pos
        if val is not None:
            vals = val
    else:
        if res is not None:
            res_scale = tf.maximum(res[::-1] // stride, 1)
            dilated_pos = grid_pos(pos,
                                   res_scale,
                                   centralize=centralize,
                                   pad=pad,
                                   hyst=hyst)
            pcnt = tf.shape(pos)[0]
        else:
            sample_cnt = tf.maximum(tf.shape(pos)[0] // stride, 1)
            pcnt = sample_cnt
            idx = farthest_point_sample(sample_cnt, tf.expand_dims(pos,
                                                                   axis=0))
            dilated_pos = gather_point(tf.expand_dims(pos, axis=0), idx)[0]
            if val is not None:
                vals = gather_point(tf.expand_dims(val, axis=0), idx)[0]
    return dilated_pos, pcnt, vals


def get_dilated_pos(pos,
                    strides,
                    voxel_size=None,
                    centralize=False,
                    pad=0,
                    hyst=0.1):
    pcnt = []
    dilated_pos = []
    idx = []

    for stride in strides:
        if stride == 1:
            pcnt.append(tf.shape(pos)[0])
            dilated_pos.append(pos)
            idx.append(None)
        else:
            if voxel_size is not None:
                v_scale = voxel_size * stride
                dilated_pos.append(
                    grid_pos(pos,
                             v_scale,
                             centralize=centralize,
                             pad=pad,
                             hyst=hyst))
                pcnt.append(tf.shape(dilated_pos[-1])[0])
            else:
                sample_cnt = tf.maximum(tf.shape(pos)[0] // stride, 1)
                pcnt.append(sample_cnt)
                idx.append(
                    farthest_point_sample(
                        sample_cnt, tf.expand_dims(dilated_pos[-1], axis=0)))
                dilated_pos.append(
                    gather_point(tf.expand_dims(dilated_pos[-1], axis=0),
                                 idx[-1])[0])

    return dilated_pos, pcnt, idx


def cubic_spline_kernel_3d(h, *, r=None, q=None):
    # 计算sigma的值
    sigma = 8 / (np.pi * h ** 3)

    # 检查输入参数
    if q is None and r is not None:
        q = r / h
    elif q is None and r is None:
        raise ValueError("Either r or q must be provided")

    # 定义条件
    cond1 = tf.logical_and(tf.greater_equal(q, 0), tf.less(q, 0.5))
    cond2 = tf.logical_and(tf.greater_equal(q, 0.5), tf.less(q, 1.0))

    # 计算每个条件下的值
    result1 = sigma * (6 * (q ** 3 - q ** 2) + 1)
    result2 = sigma * (2 * (1 - q ** 3))

    # 根据条件选择结果
    result = tf.where(cond1, result1, tf.where(cond2, result2, tf.zeros_like(q)))

    return result


def compute_density(out_pos, in_pos=None, radius=0.005, mass=None, nns=None, ignore_neighbors_grad=False):
    return compute_kernel_sum(out_pos, radius, in_pos, mass, nns, ignore_neighbors_grad)


def compute_kernel_sum(out_pos, radius, in_pos=None, in_mass=None, nns=None, ignore_neighbors_grad=False):
    if in_pos is None:
        in_pos = out_pos

    radius = tf.convert_to_tensor(radius)
    if nns is not None:
        neighbors_index, neighbors_row_splits, dist = nns
    else:
        fixed_radius_search = ml3d.layers.FixedRadiusSearch()
        neighbors_index, neighbors_row_splits, dist = fixed_radius_search(in_pos, out_pos, radius)

    neighbors = tf.RaggedTensor.from_row_splits(
        values=tf.gather(in_pos, neighbors_index),
        row_splits=neighbors_row_splits)

    if ignore_neighbors_grad:
        neighbors = tf.stop_gradient(neighbors)

    # 计算 RaggedTensor 的范数
    squared_dist = tf.reduce_sum(tf.square(neighbors - tf.expand_dims(out_pos, axis=1)), axis=-1)
    dist = tf.sqrt(squared_dist)

    if in_mass is None:
        add_sum = tf.reduce_sum(cubic_spline_kernel_3d(radius, r=dist), axis=-1)
    else:
        neighbors_mass = tf.RaggedTensor.from_row_splits(
            values=tf.gather(in_mass, neighbors_index),
            row_splits=neighbors_row_splits)
        add_sum = tf.reduce_sum(cubic_spline_kernel_3d(radius, r=dist) * neighbors_mass[..., -1], axis=-1)
    return add_sum


def quat_mult(q, r):
    return tf.stack([
        r[..., 0] * q[..., 0] - r[..., 1] * q[..., 1] - r[..., 2] * q[..., 2] -
        r[..., 3] * q[..., 3], r[..., 0] * q[..., 1] + r[..., 1] * q[..., 0] -
        r[..., 2] * q[..., 3] + r[..., 3] * q[..., 2], r[..., 0] * q[..., 2] +
        r[..., 1] * q[..., 3] + r[..., 2] * q[..., 0] - r[..., 3] * q[..., 1],
        r[..., 0] * q[..., 3] - r[..., 1] * q[..., 2] + r[..., 2] * q[..., 1] +
        r[..., 3] * q[..., 0]
    ],
        axis=-1)


def quat_conj(q):
    return q * [1, -1, -1, -1]


def quat_rot(v, q):
    r = tf.concat([tf.zeros_like(v[..., :1]), v], axis=-1)
    return quat_mult(quat_mult(q, r), quat_conj(q))[..., 1:]


def quat_mean(q0, q1):
    return (q0 + q1) / tf.expand_dims(
        tf.sqrt(2 + 2 * tf.reduce_sum(q0 * q1, axis=-1)), axis=-1)


def compute_transformed_dx(pos, scale=None, rot=None, radius=0.005):
    radius = tf.convert_to_tensor(radius)
    fixed_radius_search = ml3d.layers.FixedRadiusSearch()
    neighbors_index, neighbors_row_splits, dist = fixed_radius_search(
        pos, pos, radius)

    neighbors = tf.RaggedTensor.from_row_splits(
        values=tf.gather(pos, neighbors_index),
        row_splits=neighbors_row_splits)

    dx = neighbors - tf.expand_dims(pos, axis=1)

    if rot is not None:
        neighbors = tf.RaggedTensor.from_row_splits(
            values=tf.gather(rot, neighbors_index),
            row_splits=neighbors_row_splits)
        rot = quat_mean(neighbors, tf.expand_dims(rot, axis=1))
        dx = quat_rot(dx, rot)

    if scale is not None:
        scale = tf.RaggedTensor.from_row_splits(
            values=tf.gather(scale, neighbors_index),
            row_splits=neighbors_row_splits)
        # scale = (neighbors + tf.expand_dims(scale, axis=1)) / 2
        dx = dx * scale

    dx = tf.reduce_mean(dx, axis=1)
    return dx


def compute_pressure(out_pts,
                     inp_pts=None,
                     dens=None,
                     rest_dens=3.5,
                     stiffness=20.0,
                     win=get_window_func('poly6')):
    if inp_pts is None:
        inp_pts = out_pts
    raise NotImplementedError('compute_pressure')
    dens = compute_density(out_pts, inp_pts) if dens is None else dens
    pres = tf.keras.activations.relu(stiffness * ((dens / rest_dens) ** 7 - 1))
    return pres


def density_loss(gt,
                 pred,
                 gt_in=None,
                 pred_in=None,
                 radius=0.005,
                 eps=0.01,
                 win=None,
                 use_max=False,
                 **kwargs):
    pred_dens = compute_density(pred, pred_in, radius)
    gt_dens = compute_density(gt, gt_in, radius)

    rest_dens = tf.math.reduce_max(gt_dens)

    if use_max:
        return tf.abs(tf.reduce_max(pred_dens) - rest_dens) / rest_dens

    err = tf.keras.activations.relu(pred_dens - rest_dens - eps)
    return tf.reduce_mean(err)


def density_loss_pbf(label, pos, pred_dens, density0, fac, eps=0.01, use_max=False, **kwargs):
    rest_dens = density0
    if use_max:
        return tf.abs(tf.reduce_max(pred_dens) - rest_dens) / rest_dens
    err = (pred_dens / rest_dens - 1) ** 2
    return tf.reduce_mean(err) * fac


def emd_loss(y_true, y_pred, n=None, m=None):
    if n is None:
        n = tf.tile(tf.shape(y_true)[1:2], tf.shape(y_true)[:1])
    if m is None:
        m = tf.tile(tf.shape(y_pred)[1:2], tf.shape(y_pred)[:1])
    match = approx_match(y_true, y_pred, n, m)
    return match_cost(y_true, y_pred, match) / tf.cast(tf.maximum(n, m),
                                                       tf.float32)


def approx_vel(pos_0, pos_1, n=None, m=None):
    vel = tf.expand_dims(pos_1, axis=2) - tf.expand_dims(pos_0, axis=1)
    match = tf.expand_dims(approx_match(pos_0, pos_1, n, m), axis=-1)
    return tf.reduce_sum(vel * match, axis=1)
