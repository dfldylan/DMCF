# SpheroNet

![TensorFlow badge](https://img.shields.io/badge/TensorFlow-supported-brightgreen?style=flat&logo=tensorflow)

This repository contains the code for SpheroNet. Our algorithm makes it possible to learn highly accurate, efficient and momentum conserving fluid simulations based on particles.
With the code published here, evaluations from the paper can be reconstructed, and new models can be trained.

## Dependencies and Setup

Used environment: python 3.8.10 with CUDA 11.2 and CUDNN 8.1.
- Install libcap-dev: ```sudo apt install libcap-dev```
- Install cmake: ```sudo apt install cmake```
- Update pip: ```pip install --upgrade pip```
- Install requirements: ```pip install -r requirements.txt```

Optional: 
- Build FPS/EMD module ```cd utils; make; cd ..```
- Install skia for visualization: ```python -m pip install skia-python```

## Datasets

- *WBC-SPH*: https://dataserv.ub.tum.de/index.php/s/m1693614
- *Liquid3d* ([source](https://github.com/isl-org/DeepLagrangianFluids)): https://drive.google.com/file/d/1b3OjeXnsvwUAeUq2Z0lcrX7j9U7zLO07
- *WaterRamps* ([source](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)): ```bash download_waterramps.sh PATH/TO/OUTPUT_DIR```

### How to set dataset path

Under configs/*.yml:

```yaml
dataset:
  dataset_path:  # path to dataset
```

## Path to runtime generated data

Under configs/*.yml:

```yaml
pipeline:
  base_data_dir : # path to output data
  
  main_log_dir: ./logs  # os.path.join(base_data_dir, main_log_dir)
  train_sum_dir: ./train_log  # os.path.join(base_data_dir, train_sum_dir)
  output_dir: ./output  # os.path.join(base_data_dir, output_dir)
```

## Pretrained Models:

The pretrained models are in the ```checkpoints``` subfolder.
Run a pretrained mode by setting the path to the checkpoint with the ```ckpt_path``` argument.
For example:
```bash
python run_pipeline.py --cfg_file WBC-SPH.yml \
                       --ckpt_path checkpoints/WBC-SPH/ckpt \
                       --split test
```

## Training

Simple 1D test run (data will be generated):
```bash
python run_pipeline.py --cfg_file column/hrnet.yml \
                       --split train
```

Run with 2D pipeline:
```bash
python run_pipeline.py --cfg_file WBC-SPH.yml \
                       --split train
```

Run with 3D pipeline:
```bash
python run_pipeline.py --cfg_file Liquid3d.yml \
                       --split train
```

## Test

```bash
python run_pipeline.py --cfg_file WBC-SPH.yml \
                       --split test \
                       --pipeline.data_generator.test.time_end 800 \
                       --pipeline.data_generator.valid.time_end 800 \
                       --pipeline.data_generator.valid.random_start 0 \
                       --pipeline.test_compute_metric true
```
*Note: The argument ```pipeline.data_generator.test.time_end```, ```pipeline.data_generator.valid.time_end```, ```pipeline.data_generator.valid.random_start```, and ```pipeline.test_compute_metric``` are examples how to overwrite corresponding entries in the config file.*

The ```...time_end``` parameter account for the number of frames used for inference and evaluation. We used a value of *3200* for the *WBC-SPH* data set, *600* for *WaterRamps*, and *200* for *Liquid3d*.
The generated test files are stored in the ```pipeline.output_dir``` folder, specified in the config file. The output files have a *hdf5* format and can be rendered with the ```utils/draw_sim2d.py``` script.

Rendering of a small sample sequence:
```bash
python utils/draw_sim2d.py PATH/TO/HDF5_FILE OUTPUT/PATH
```

Rendering of individual frames:
```bash
python utils/draw_sim2d.py PATH/TO/HDF5_FILE OUTPUT/PATH \
                           --out_pattern OUTPUT/FRAMES/{frame:04d}.png \
                           --num_frames 800
```

## Validation

```bash
python run_pipeline.py --cfg_file WBC-SPH.yml \
                       --split valid
```

## Licenses
Code and scripts are under the MIT license.

Data files are under the CDLA-Permissive-2.0 license.
