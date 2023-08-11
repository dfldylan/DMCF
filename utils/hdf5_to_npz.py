import argparse
import os, sys, shutil
import numpy as np
import yaml
import h5py


def parse_args():
    parser = argparse.ArgumentParser(description='Convert hdf5 to a series of npz')
    parser.add_argument('-s', '--source_file', help='path to the hdf5 file')
    parser.add_argument('-d', '--output_dir', help='the dir to save outputs')

    args, unknown = parser.parse_known_args()

    parser_extra = argparse.ArgumentParser(description='Extra arguments')
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser_extra.add_argument(arg)
    args_extra = parser_extra.parse_args(unknown)

    print("regular arguments")
    print(yaml.dump(vars(args)))

    print("extra arguments")
    print(yaml.dump(vars(args_extra)))

    return args, vars(args_extra)


def reset_folder(path):
    # 如果文件夹存在，则删除整个文件夹
    if os.path.exists(path):
        shutil.rmtree(path)
    # 重新创建文件夹
    os.makedirs(path)


def save_npz(folder, name, *args, **kwargs):
    np.savez(os.path.join(folder, str(name) + '.npz'), **kwargs)


def write_npz(dst, data):
    gt_folder = os.path.join(dst, 'gt_npz')
    pred_folder = os.path.join(dst, 'pred_npz')
    reset_folder(gt_folder)
    reset_folder(pred_folder)
    save_npz(gt_folder, 'box', box=data['bnd'])
    save_npz(pred_folder, 'box', box=data['bnd'])
    for i in range(data['gt'].shape[0]):
        save_npz(gt_folder, 'fluid_{0:04d}'.format(i), pos=data['gt'][i])
    for i in range(data['pred'].shape[0]):
        save_npz(pred_folder, 'fluid_{0:04d}'.format(i), pos=data['pred'][i])
    return True


def convert_hdf5_npz(src, dst):
    with h5py.File(src, 'r') as h:
        key = 'PolarNet'
        if not key in h and len(h) == 1:
            key = next(iter(h))
        data = {k: v[:] for k, v in h[key].items()}
    return write_npz(dst, data)


def main():
    args, extra_dict = parse_args()
    if args.source_file is not None:
        src = args.source_file
        dst = args.output_dir if args.output_dir is not None else os.path.dirname(src)
        os.makedirs(dst, exist_ok=True)
        flag = convert_hdf5_npz(src, dst)
    return flag


if __name__ == '__main__':
    main()
