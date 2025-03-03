import random

import tensorflow as tf
import logging
import numpy as np
from tqdm import tqdm
import re
import os
import time
from glob import glob
import time

from datetime import datetime

from os.path import exists, join
from pathlib import Path

from .base_pipeline import BasePipeline

from o3d.utils import make_dir, PIPELINE, LogRecord, get_runid, code2md

from datasets.dataset_reader_physics import get_dataloader, get_rollout, write_results

from utils.tools.losses import density_loss, get_window_func, compute_density, get_window_func, emd_loss, boundary_loss
from utils.evaluation_helper import compare_dist, chamfer_distance, distance, merge_dicts
from utils.hdf5_to_npz import write_npz
import warnings

warnings.filterwarnings('ignore')

logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Simulator(BasePipeline):
    """
    Pipeline for trainable simulator. 
    """

    def __init__(self,
                 model,
                 dataset=None,
                 name='Simulator',
                 main_log_dir='./logs/',
                 device='cuda',
                 split='train',
                 **kwargs):
        super().__init__(model=model,
                         dataset=dataset,
                         name=name,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         **kwargs)

    # @tf.function(experimental_relax_shapes=True)
    def run_inference(self, inputs):
        """
        Run inference on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """
        results = []
        for bi in range(len(inputs)):
            pos, vel = self.model(inputs[bi], training=False)
            results.append([pos, vel] + inputs[bi][2:])
        return results

    def run_rollout(self, inputs, timesteps=2):
        """
        Run rollout on a given data.

        Args:
            data: A raw data.
        Returns:
            Returns the inference results.
        """

        inputs = [[
            tf.convert_to_tensor(data['pos'][0]),
            tf.convert_to_tensor(data['vel'][0]),
            tf.convert_to_tensor(data["grav"][0])
            if data["grav"][0] is not None else None, None,
            tf.convert_to_tensor(data["box"][0]),
            tf.convert_to_tensor(data["box_normals"][0])
        ] for data in inputs]
        results = [[] for _ in range(len(inputs))]

        # dummy init
        self.run_inference(inputs[:1])

        timing = []
        for i in range(len(inputs)):
            results[i].append(inputs[i])
        log.info("rollout total:", str(timesteps))
        for t in tqdm(range(timesteps - 1), "rollout"):
            start = time.time()
            for i in range(len(inputs)):
                inputs[i] = self.run_inference(inputs[i:i + 1])[0]
            end = time.time()
            timing.append(end - start)
            for i in range(len(inputs)):
                results[i].append(inputs[i])
        log.info("Average runtime: %.05f" % (np.mean(timing) / len(inputs)))

        return results

    def run_test(self, epoch=None, test_dataset=None):
        """
        Run test with test data split.
        """
        model = self.model
        dataset = self.dataset

        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        if not test_dataset:
            test_dataset = dataset.test
        test_data = get_rollout(test_dataset, **cfg.data_generator,
                                **cfg.data_generator.test)

        if epoch is None:
            epoch = self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started testing")

        results = self.run_rollout(test_data, test_data[0]["pos"].shape[0])

        for i in tqdm(range(len(results)), desc='write out'):
            data = test_data[i]
            pos = np.stack(r[0] for r in results[i])

            out_dir = os.path.join(self.cfg.out_dir, "visual", "%04d" % i)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            output = [(pos, {
                "name": "pred",
                "type": "PARTICLE"
            }), (data['pos'], {
                "name": "gt",
                "type": "PARTICLE"
            }), (data['box'][0], {
                "name": "bnd",
                "type": "PARTICLE"
            })]

            write_results(os.path.join(out_dir, '%04d.hdf5' % epoch),
                          self.model.name, output)

            write_npz(out_dir, data={'pred': pos, 'gt': data['pos'], 'bnd': data['box'][0]})

            for f in glob(os.path.join(out_dir, '*.hdf5')):
                if f != os.path.join(out_dir, '%04d.hdf5' % epoch):
                    log.info("Remove %s" %
                             os.path.join(out_dir, '%04d.hdf5' % epoch))
                    os.remove(f)

        if cfg.get('test_compute_metric', False):
            self.run_valid(epoch)

    def run_valid(self, epoch=None):
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        log_file_path = join(cfg.logs_dir, 'log_valid_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))

        valid_data = get_rollout(dataset.valid, **cfg.data_generator,
                                 **cfg.data_generator.valid)  # [batch, timesteps, 6]

        if epoch is None:
            epoch = self.load_ckpt(model.cfg.ckpt_path)

        log.info("Started validation")

        results = self.run_rollout(valid_data, valid_data[0]["pos"].shape[0])  # [batch, timesteps, 6]

        losses = []
        for i in tqdm(range(len(valid_data)), desc='validation'):
            data = valid_data[i]
            target_pos, target_vel = data["pos"], data["vel"]

            loss_seq = []
            for t in range(1, min(target_pos.shape[0], len(results[i]))):  # for every particle
                # eval for complete sequence
                pos, vel = results[i][t][:2]
                loss = {}

                if t % cfg.data_generator.valid.get("eval_stride", 1) == 0:
                    if data["box"][0].shape[0] > 0:
                        pos = tf.clip_by_value(pos,
                                               np.min(data["box"][0], axis=0),
                                               np.max(data["box"][0], axis=0))
                    loss['mse_val'] = np.mean(distance(target_pos[t], pos))
                    loss['chamfer_val'] = np.mean(
                        chamfer_distance(target_pos[t],
                                         pos).astype(np.float32))

                    if cfg.split != "train":
                        loss['dens_val'] = np.mean(
                            density_loss(target_pos[t],
                                         pos,
                                         tf.concat([pos, data["box"][0]],
                                                   axis=0),
                                         tf.concat(
                                             [target_pos[t], data["box"][0]],
                                             axis=0),
                                         radius=model.particle_radii[0],
                                         win=get_window_func("poly6")).numpy())
                        loss['max_dens_val'] = density_loss(
                            pos,
                            target_pos[t],
                            tf.concat([pos, data["box"][0]], axis=0),
                            tf.concat([target_pos[t], data["box"][0]], axis=0),
                            radius=model.particle_radii[0],
                            win=get_window_func(model.window_dens),
                            use_max=True).numpy()
                        loss['chamfer_val_2'] = np.mean(
                            chamfer_distance(pos,
                                             target_pos[t]).astype(np.float32))
                        loss['emd'] = np.mean(
                            emd_loss(target_pos[t:t + 1], tf.expand_dims(
                                pos, 0)).numpy().astype(np.float32))

                        loss['vel_diff_val'] = compare_dist(target_vel[t], vel)
                        loss['vel_diff_val_2'] = compare_dist(
                            vel, target_vel[t])

                        loss['weight_mse_val'] = boundary_loss(target_pos[t], pos, data['box'],
                                                               radius=model.particle_radii[0])
                        np.mean(distance(target_pos[t], pos))

                    # mse for single step only
                    try:
                        pos_sub = self.model(
                            [target_pos[t - 1], target_vel[t - 1]] +
                            results[i][t][2:])[0]
                    except tf.errors.ResourceExhaustedError as e:
                        logging.info(f"ResourceExhaustedError: {e.message}")
                        tf.keras.backend.clear_session()
                        tf.compat.v1.reset_default_graph()
                        continue

                    loss['mse_single_val'] = np.mean(
                        distance(target_pos[t], pos_sub))

                    losses.append(loss)
                    loss_seq.append(loss)

            loss_m = merge_dicts(loss_seq, lambda x, y: x + y / len(loss_seq))

            desc = "%d -" % i
            for l, v in loss_m.items():
                desc += " %s: %.05f" % (l, v)

            log.info(desc)

        loss = merge_dicts(losses, lambda x, y: x + y / len(losses))

        sum_loss = 0
        desc = "validation of epoch %d -" % epoch
        for l, v in loss.items():
            desc += " %s: %.05f" % (l, v)
            sum_loss += v
        desc += " > loss: %.05f" % sum_loss
        loss["loss"] = sum_loss

        log.info(desc)

        self.valid_loss = loss

    def run_train(self):
        model = self.model
        dataset = self.dataset
        cfg = self.cfg

        # 设置日志文件路径
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path = join(cfg.logs_dir, f'log_train_{timestamp}.txt')
        log.info(f"Logging in file: {log_file_path}")
        log.addHandler(logging.FileHandler(log_file_path))

        # 获取训练数据加载器
        train_loader = get_dataloader(
            dataset.train,
            batch_size=cfg.batch_size,
            pre_frames=cfg.max_warm_up[0],
            max_pre_frames=cfg.max_warm_up[-1],
            window=cfg.windows[0],
            max_window=cfg.windows[-1],
            **cfg.data_generator,
            **cfg.data_generator.train
        )

        self.optimizer = model.get_optimizer(cfg.optimizer)
        is_resume = model.cfg.get('is_resume', True)
        start_epoch = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        log.info(f"Writing summary in {self.tensorboard_dir}.")

        window_idx, warmup_idx, iteration_idx = 0, 0, 0
        log.info("Started training")

        error_samples = []
        for epoch in range(start_epoch, cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch}/{cfg.max_epoch} ===')
            process_bar = tqdm(range(cfg.iter), desc='training')
            # 动态计算 target_loss
            target_loss = cfg.optimizer.loss_values[-1]
            for i in range(len(cfg.optimizer.loss_boundaries)):
                if epoch < (cfg.optimizer.loss_boundaries[i] // cfg.iter):
                    target_loss = cfg.optimizer.loss_values[i]
                    break
            for iteration in process_bar:
                step = epoch * cfg.iter + iteration
                if iteration < 0.8 * cfg.iter:

                    loader_updated = False

                    while window_idx < min(len(cfg.windows), len(cfg.window_bnds)) and step >= cfg.window_bnds[
                        window_idx]:
                        window_idx += 1
                        loader_updated = True

                    while warmup_idx < min(len(cfg.max_warm_up), len(cfg.warm_up_bnds)) and step >= cfg.warm_up_bnds[
                        warmup_idx]:
                        warmup_idx += 1
                        loader_updated = True

                    if loader_updated:
                        train_loader = get_dataloader(
                            dataset.train,
                            batch_size=cfg.batch_size,
                            pre_frames=cfg.max_warm_up[warmup_idx],
                            window=cfg.windows[window_idx],
                            **cfg.data_generator,
                            **cfg.data_generator.train
                        )
                        # time.sleep(10)

                    data_fetch_start = time.time()
                    data = next(train_loader)
                    time_weights = self.calculate_time_weights(cfg, data, step, window_idx)

                    data_fetch_latency = time.time() - data_fetch_start
                else:
                    data_fetch_start = time.time()
                    data = random.choice(error_samples) if error_samples else next(train_loader)
                    time_weights = self.calculate_time_weights(cfg, data, step, window_idx)

                    data_fetch_latency = time.time() - data_fetch_start
                self.log_scalar_every_n_minutes(self.writer, step, 5, 'DataLatency', data_fetch_latency)

                try:
                    loss, pre_steps = self.train_step(model, cfg, self.optimizer, data, time_weights)
                except tf.errors.ResourceExhaustedError as e:
                    logging.info(f"ResourceExhaustedError: {e.message}")
                    tf.keras.backend.clear_session()
                    tf.compat.v1.reset_default_graph()
                    continue

                if iteration == 0 and epoch == start_epoch:
                    self.log_param_count()

                # 在主代码中使用新的函数
                loss_values, desc = self.log_and_generate_description(model, loss, time_weights, data, pre_steps)
                if iteration < 0.8 * cfg.iter and loss_values['loss'] >= target_loss:
                    if not any(data is item for item in error_samples):
                        logging.info("loss: {} >= target_loss: {}, add to error_samples".format(loss_values['loss'],
                                                                                                target_loss))
                        error_samples.append(data)
                elif iteration >= 0.8 * cfg.iter and loss_values['loss'] < target_loss:
                    error_samples[:] = [item for item in error_samples if data is not item]
                    logging.info("loss: {} < target_loss: {}, remove from error_samples".format(loss_values['loss'],
                                                                                                target_loss))
                process_bar.set_description(desc)
                process_bar.refresh()

                self.save_logs(self.writer, step, [loss_values], "train")

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

            # 运行验证和测试
            # --------------------- validation
            self.run_valid(epoch)
            self.save_logs(self.writer, epoch, [self.valid_loss], "valid")

            self.run_test(epoch)

    def calculate_time_weights(self, cfg, data, step, window_idx):
        time_weights = np.ones((np.min([d.shape[0] - 1 - p for d, p in zip(data['pos'], data['pre'])])),
                               dtype=np.float32)
        if window_idx > 0:
            alpha = (step - cfg.window_bnds[window_idx - 1] + 1) / cfg.time_blend
            if alpha < 1.0 and len(time_weights) >= cfg.windows[window_idx]:
                diff = cfg.windows[window_idx] - cfg.windows[window_idx - 1]
                time_weights[-diff:] = np.clip(alpha - np.arange(diff) / diff, 0.0, 1.0)
        return tf.convert_to_tensor(list(time_weights))

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, model, cfg, optimizer, data, time_weights):
        in_positions, in_velocities, pre_steps = self.warmup_phase(model, data, cfg)
        total_loss = self.calculate_loss(model, cfg, optimizer, data, in_positions, in_velocities, pre_steps,
                                         time_weights)
        return total_loss, pre_steps

    def warmup_phase(self, model, data, cfg):
        in_positions, in_velocities, pre_steps = [], [], []
        for batch_index in range(len(data['pos'])):
            acc = data["grav"][batch_index][0]
            pr_pos = data["pos"][batch_index][0]
            pr_vel = data["vel"][batch_index][0]
            step, prev_err, prev_dens_err = 0, 0.0, 0.0

            for step in range(data['pre'][batch_index]):
                inputs = (pr_pos, pr_vel, acc, None, data["box"][batch_index][0], data["box_normals"][batch_index][0])
                pos, vel = model(inputs, training=False)
                signal, prev_err, prev_dens_err = self.check_error_thresholds(pos, data, batch_index, step, prev_err,
                                                                              prev_dens_err, cfg, model)
                if not signal:
                    break
                pr_pos, pr_vel = pos, vel
            pre_steps.append(step)
            in_positions.append(pr_pos)
            in_velocities.append(pr_vel)
        return in_positions, in_velocities, pre_steps

    def check_error_thresholds(self, pos, data, batch_index, step, prev_err, prev_dens_err, cfg, model):
        max_err = cfg.get('max_err', None)
        max_dens_err = cfg.get('max_dens_err', None)
        if max_err is not None:
            err = tf.reduce_max(tf.reduce_sum(tf.abs(pos - data["pos"][batch_index][step]), axis=-1))
            if step > 0 and err > prev_err and err > max_err:
                return False, prev_err, prev_dens_err
            prev_err = err
        if max_dens_err is not None:
            err = density_loss(
                pos,
                data["pos"][batch_index][step],
                tf.concat([pos, data["box"][batch_index][0]], axis=0),
                tf.concat([data["pos"][batch_index][step], data["box"][batch_index][0]], axis=0),
                radius=model.particle_radii[0],
                win=get_window_func(model.window_dens),
                use_max=True)
            if step > 0 and err > prev_dens_err and err > max_dens_err:
                return False, prev_err, prev_dens_err
            prev_dens_err = err
        return True, prev_err, prev_dens_err

    def calculate_loss(self, model, cfg, optimizer, data, in_positions, in_velocities, pre_steps, time_weights):
        with tf.GradientTape() as tape:
            loss_tensor_array = tf.TensorArray(tf.float32, size=tf.shape(time_weights)[0] * len(data['pos']),
                                               dynamic_size=True, clear_after_read=False)

            for batch_index in range(len(data['pos'])):
                pos, vel = in_positions[batch_index], in_velocities[batch_index]
                pre = pre_steps[batch_index]
                t = 0

                while t < tf.shape(time_weights)[0]:
                    pos, vel, pre, t, loss_tensor_array = self.train_step_body(pos, vel, pre, t, loss_tensor_array,
                                                                               model, data, batch_index, time_weights)

            total_loss = tf.reduce_sum(loss_tensor_array.stack(), axis=0) / (
                    tf.reduce_sum(time_weights) * len(data['pos']))
            total_loss = self.apply_weight_decay(cfg, model, total_loss)
            gradients = tape.gradient(total_loss, model.trainable_weights)
            self.apply_gradients(optimizer, cfg, gradients, model)
        return total_loss

    def train_step_body(self, pos, vel, pre, t, loss_array, model, data, batch_index, time_weights):
        inputs = [pos, vel, data["grav"][batch_index][0], None, data["box"][batch_index][0],
                  data["box_normals"][batch_index][0]]
        target_pos = data["pos"][batch_index]
        target_vel = data["vel"][batch_index]
        pos, vel = model(inputs, training=True)
        loss_list = [model.loss([pos, vel],
                                [inputs, target_pos[t + pre + 1], target_pos[t + pre], pre])]
        merged_loss = merge_dicts(loss_list, lambda x, y: x + y / len(loss_list))
        loss_array = loss_array.write(t + batch_index * tf.shape(time_weights)[0],
                                      tf.convert_to_tensor(list(merged_loss.values())) * time_weights[t])
        return pos, vel, pre, t + 1, loss_array

    def apply_weight_decay(self, cfg, model, total_loss):
        weight_decay = cfg.get("w_decay", 0)
        if weight_decay > 0:
            total_loss += weight_decay * tf.reduce_sum(
                [tf.reduce_sum(weight ** 2) for weight in model.trainable_weights])
        return total_loss

    def apply_gradients(self, optimizer, cfg, gradients, model):
        grad_clip_norm = cfg.get('grad_clip_norm', -1)
        if grad_clip_norm > 0:
            gradients = [tf.clip_by_norm(grad, grad_clip_norm) for grad in gradients]
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    def log_and_generate_description(self, model, loss, time_weights, data, pre_steps):
        loss_values = {}
        total_loss = np.sum(loss)
        loss_values["loss"] = total_loss
        loss_values["timesteps"] = np.minimum(tf.reduce_sum(time_weights).numpy(),
                                              tf.cast(tf.math.ceil(tf.reduce_sum(time_weights)), tf.int32).numpy())
        loss_values["warmup"] = tf.reduce_mean(data['pre'])
        loss_values["warmup_diff"] = tf.reduce_mean(tf.convert_to_tensor(data['pre']) - tf.convert_to_tensor(pre_steps))

        desc = "training -"
        for loss_key, loss_val in zip(model.loss_keys(), loss):
            loss_values[loss_key] = loss_val.numpy()
            desc += f" {loss_key}: {loss_val.numpy():.05f}"
        desc += f" > loss: {total_loss:.05f}"

        return loss_values, desc
