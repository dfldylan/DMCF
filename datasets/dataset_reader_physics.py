import os
import sys
from datetime import datetime
from glob import glob
import tensorflow as tf
import numpy as np
import zstandard as zstd
import collections
import msgpack
import msgpack_numpy
import shutil

msgpack_numpy.patch()
import h5py
import random
from collections import deque
from multiprocessing import Process, Manager
from queue import Empty

from datasets import column_gen, free_fall_gen

Stats = collections.namedtuple('Stats', ['mean', 'std'])

from scipy.spatial.transform import Rotation as R

import hashlib
import json


def dict_hash(d):
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(d, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def align_vector(v0, v1):
    v0_norm = v0 / (np.linalg.norm(v0) + 1e-9)
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-9)

    v = np.cross(v0_norm, v1_norm)
    c = np.dot(v0_norm, v1_norm)
    s = np.linalg.norm(v)

    if s < 1e-6:
        return np.eye(3).astype(np.float32) * (-1.0 if c < 0 else 1.0)

    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    r = np.eye(3) + vx + np.dot(vx, vx) / (1 + c)
    return r.astype(np.float32)


def random_rotation_matrix(rot_axis=None, dtype=np.float32):
    """Generates a random rotation matrix 
    
    strength: scalar in [0,1]. 1 generates fully random rotations. 0 generates the identity. Default is 1.
    dtype: output dtype. Default is np.float32
    """

    x = np.random.rand(3)
    theta = x[0] * 2 * np.pi

    st = np.sin(theta)
    ct = np.cos(theta)

    if rot_axis is not None:
        if rot_axis == 0:
            return np.array([[1, 0, 0], [0, ct, st], [0, -st,
                                                      ct]]).astype(dtype)
        elif rot_axis == 1:
            return np.array([[ct, 0, st], [0, 1, 0], [-st, 0,
                                                      ct]]).astype(dtype)
        else:
            return np.array([[ct, st, 0], [-st, ct, 0], [0, 0,
                                                         1]]).astype(dtype)

    phi = x[1] * 2 * np.pi
    z = x[2] * strength

    r = np.sqrt(z)
    V = np.array([np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)])

    Rz = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])
    rand_R = (np.outer(V, V) - np.eye(3)).dot(Rz)
    return rand_R.astype(dtype)


class DatasetGroup:
    def __init__(self,
                 train=None,
                 valid=None,
                 test=None,
                 split="train",
                 regen=False,
                 **dataset_cfg):
        self.name = dataset_cfg.pop("name")
        if 'dataset_path' not in dataset_cfg:
            # no path => generate data

            gen_type = dataset_cfg.pop("type", "tank")
            if gen_type == "tank":
                # TODO
                raise NotImplementedError()
                # f = tank.gen_data
            elif gen_type == "column":
                f = column_gen.gen_data
            elif gen_type == "free_fall":
                f = free_fall_gen.gen_data
            else:
                raise NotImplementedError()

            print("gen train data")
            self.train = self.gen_data(f, regen=regen, **train, **dataset_cfg)
            print("gen valid data")
            self.valid = self.gen_data(f, regen=regen, **valid, **dataset_cfg)
            print("gen test data")
            self.test = self.gen_data(f, regen=regen, **test, **dataset_cfg)
        else:
            path = dataset_cfg.pop("dataset_path")

            if split == "train":
                if not os.path.exists(os.path.join(path, "train")):
                    raise FileNotFoundError()
                self.train = Dataset(dataset_path=os.path.join(path, "train"),
                                     **dataset_cfg)

            if os.path.exists(os.path.join(path, "valid")):
                self.valid = Dataset(dataset_path=os.path.join(path, "valid"), **dataset_cfg)
            else:
                self.valid = Dataset(dataset_path=path, **dataset_cfg)

            if split != "valid":
                if os.path.exists(os.path.join(path, "test")):
                    self.test = Dataset(dataset_path=os.path.join(
                        path, "test"), **dataset_cfg)
                else:
                    try:
                        self.test = Dataset(dataset_path=path, **dataset_cfg)
                    except AssertionError:
                        self.test = self.valid

                if split == "test":
                    self.valid = self.test

    def gen_data(self, func, regen=False, **cfg):
        fldr = dict_hash(cfg)
        cache_dir = os.path.join("cache", fldr)
        seed = None
        if "seed" in cfg:
            seed = cfg.pop("seed")
            np.random.seed(seed)
            if regen and os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            if os.path.exists(cache_dir):
                print("found cache, load from cache: %s" % cache_dir)
                decompressor = zstd.ZstdDecompressor()
                with open(os.path.join(cache_dir, "data.msgpack.zst"),
                          'rb') as f:
                    data = msgpack.unpackb(decompressor.decompress(f.read()),
                                           raw=False)
                data = Dataset(data)
                return data

        print("no cache found or seed not set, generate data")
        data = func(**cfg)
        data = Dataset(data)

        if seed is not None:
            print("data generated, store in cache: %s" % cache_dir)
            os.makedirs(cache_dir)
            compressor = zstd.ZstdCompressor(level=22)
            with open(os.path.join(cache_dir, "data.msgpack.zst"), 'wb') as f:
                f.write(
                    compressor.compress(
                        msgpack.packb(data.data, use_bin_type=True)))
        return data


class Dataset:
    def __init__(self, data=None, dataset_path=None):
        self.data = None
        self.files = None
        if dataset_path is not None:
            self.files = sorted(
                glob(os.path.join(dataset_path, '*.msgpack.zst')))
            assert len(self.files), "List of files must not be empty"
            print(self.files[0:20], '...' if len(self.files) > 20 else '')
        elif data is not None:
            self.data = data
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.data is not None:
            return len(self.data)

        return len(self.files)

    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]

        # read all data from file
        decompressor = zstd.ZstdDecompressor()
        with open(self.files[idx], 'rb') as f:
            return msgpack.unpackb(decompressor.decompress(f.read()),
                                   raw=False)


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)


class PhysicsSimDataFlow:
    """Data flow for msgpacks generated from SplishSplash simulations.

    Only returns data for fluids for now (position, velocity, mass, viscosity)

    """

    def __init__(self,
                 dataset,
                 shuffle=False,
                 window=1,
                 is2d=False,
                 pre_frames=0,
                 stride=1,
                 sample_cnt=None,
                 augment={},
                 translate=None,
                 scale=None,
                 grav_eqvar=None,
                 **kwargs):
        assert window >= 0
        self.dataset = dataset
        self.shuffle = shuffle
        self.window = window + 1
        self.is2d = is2d
        self.pre_frames = pre_frames
        self.stride = stride
        self.augment = augment
        self.translate = translate
        self.grav_eqvar = grav_eqvar
        self.scale = scale
        self.sample_cnt = sample_cnt
        self.rng = get_rng(self)

    def transform(self, data):
        for mode, config in self.augment.items():
            if config is None:
                config = {}
            if mode == "rotate":
                rand_R = random_rotation_matrix(**config)
                for k in ['box', 'box_normals', 'pos', 'vel']:
                    data[k] = np.matmul(data[k], rand_R)
                if data['grav'][0] is not None:
                    data[k] = np.matmul(data['grav'], rand_R)

            # elif mode == "shuffle":
            #     idx = np.arange(data['pos'].shape[1])
            #     self.rng.shuffle(idx)
            #     for k in ['pos', 'vel', 'grav']:
            #         data[k] = data[k][:, idx]

            #     idx = np.arange(data['box'].shape[1])
            #     self.rng.shuffle(idx)
            #     for k in ['box', 'box_normals']:
            #         data[k] = data[k][:, idx]

            elif mode == "jitter":
                for k, v in config.get("channels", {"pos", 1e-5}).items():
                    data[k] += self.rng.normal(scale=v, size=data[k].shape)

            elif mode == "jitter_inp":
                for k, v in config.get("channels", {"pos", 1e-5}).items():
                    data[k][0] += self.rng.normal(scale=v,
                                                  size=data[k][0].shape)

            else:
                raise NotImplementedError

        if self.translate is not None:
            data['pos'] += self.translate
            data['box'] += self.translate
        if self.scale is not None:
            data['pos'] *= self.scale
            data['box'] *= self.scale
            data['vel'] *= self.scale

            if data['grav'][0] is not None:
                data['grav'] *= self.scale

        if self.grav_eqvar is not None:
            # WARNING: assuming same gravity for all particles for one sequence
            R = align_vector(self.grav_eqvar, data['grav'][0, 0])
            data['orig_grav'] = data['grav'][0, 0]
            for k in ['box', 'box_normals', 'pos', 'vel', 'grav']:
                data[k] = np.matmul(data[k], R)

        return data

    def __iter__(self):
        # returns a list of dictioniaries, shape: (b, t, n, 3)
        files_idxs = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(files_idxs)

        for file_i in files_idxs:
            data = self.dataset[file_i]
            data_idxs = np.arange(len(data) - (self.window - 1 + self.pre_frames) * self.stride)
            assert (len(data_idxs) > 0)

            if self.sample_cnt is not None:
                if self.shuffle:
                    self.rng.shuffle(data_idxs)
                data_idxs = data_idxs[:self.sample_cnt]

            for data_i in data_idxs:
                sample = {}
                sample['pre'] = np.random.randint(self.pre_frames + 1)

                for k in ['pos', 'vel', 'grav', 'm', 'viscosity']:
                    if k in data[data_i]:
                        sample[k] = np.stack([
                            data[data_i + i * self.stride].get(
                                k, None).astype("float32")
                            for i in range(sample['pre'] + self.window)
                        ], 0)
                    else:
                        sample[k] = [None]

                for k in ['box', 'box_normals']:
                    if k in data[0]:
                        sample[k] = np.stack([
                            data[0].get(k, None).astype("float32")
                            for i in range(sample['pre'] + self.window)
                        ], 0)
                    else:
                        sample[k] = [np.empty((0, 3))]
                    sample[k] = np.reshape(sample[k], (len(sample[k]), -1, 3))

                for k in ['frame_id', 'scene_id']:
                    sample[k] = np.stack([
                        data[data_i + i * self.stride].get(k, None)
                        for i in range(sample['pre'] + self.window)
                    ], 0)

                if sample['grav'][0] is not None:
                    sample['grav'] = np.full_like(sample['vel'],
                                                  np.expand_dims(
                                                      sample['grav'],
                                                      1))  # / 100

                sample = self.transform(sample)
                yield sample


def get_rollout(dataset,
                stride=1,
                time_start=0,
                time_end=None,
                random_start=1,
                cnt=None,
                **kwargs):
    data_loader = PhysicsSimDataFlow(dataset=dataset,
                                     stride=1,
                                     window=0,
                                     max_window=0,
                                     **kwargs)
    rollout = []
    for data in data_loader:
        if data['frame_id'][0] == 0:
            if cnt is not None and len(rollout) >= cnt:
                break
            rollout.append([])
            random_off = 0
            if random_start > 1:
                random_off = np.random.randint(random_start * stride)
        if data['frame_id'][0] < time_start * stride + random_off or data['frame_id'][0] % stride != 0 or (
                time_end is not None and data['frame_id'][0] >= time_end * stride + random_off):
            continue
        rollout[-1].append(data)

    out = []
    for i in range(len(rollout)):
        merge = {}
        for k in ['pos', 'vel', 'grav', 'm', 'viscosity', 'frame_id', 'scene_id', 'box', 'box_normals']:
            l = [data[k] for data in rollout[i]]
            if len(l) == len(rollout[i]) and len(l) > 0:
                merge[k] = np.concatenate(l, 0)
            # else:
            #     print(k)
            #     print(rollout)
            #     print(l)
            #     print("hoi")
        if len(rollout[i]) > 0:
            out.append(merge)

    return out


class ShuffleIterator:
    def __init__(self, iterator, buffer_size):
        self.iterator = iterator
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.gen = self._generator()

    def _generator(self):
        for item in self.iterator:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(item)
            else:
                yield self._pop_and_append(item)
        while self.buffer:
            yield self.buffer.popleft()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def _pop_and_append(self, item):
        replace_index = random.randint(0, self.buffer_size - 1)
        replace_item = self.buffer[replace_index]
        self.buffer[replace_index] = item
        return replace_item


def repeat_iter(df):
    while True:
        yield from df


def batch_data_generator(data_iter, batch_size):
    batch = {}
    for item in data_iter:
        for key, value in item.items():
            batch.setdefault(key, []).append(value)
        if len(next(iter(batch.values()))) == batch_size:
            yield batch
            batch = {key: [] for key in batch}
    if any(batch.values()):
        yield batch


def get_dataloader(dataset, batch_size=1, window=1, repeat=False, shuffle_buffer=None, cache_data=False,
                   is2d=False, pre_frames=0, stride=1, translate=None, scale=None, augment={}, **kwargs):
    # caching makes only sense if the data is finite
    if cache_data:
        assert repeat == False
        assert not augment

    data_flow = PhysicsSimDataFlow(
        dataset=dataset,
        shuffle=bool(shuffle_buffer),
        window=window,
        is2d=is2d,
        pre_frames=pre_frames,
        stride=stride,
        translate=translate,
        scale=scale,
        augment=augment,
        **kwargs
    )

    if repeat:
        data_flow = repeat_iter(data_flow)

    data_flow = batch_data_generator(data_flow, batch_size)

    if cache_data:
        data_flow = list(data_flow)

    return data_flow


def write_results(path, name, data):
    with h5py.File(os.path.join(path), "w") as f:
        grp = f.create_group(name)
        for d, props in data:
            dset = grp.create_dataset(props['name'], data=d)
            dset.attrs["type"] = props.get("type", "DENSITY")
            dset.attrs["dim"] = d.shape
