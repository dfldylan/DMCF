import os
import numpy as np
import pandas as pd


def one_csv_to_npz(file_path, out_path, current_frame=None, box=True):
    os.makedirs(out_path, exist_ok=True)
    if not current_frame:
        try:
            current_frame = int(os.path.splitext(os.path.basename(file_path))[0].split('_')[-1])
        except:
            current_frame = 0
    df = pd.read_csv(file_path)
    df_fluid = df[df['isFluidSolid'] == 0]
    pos, vel = df_fluid.loc[:, 'position_x':'position_z'].values, df_fluid.loc[:,
                                                                  'velocity_x':'velocity_z'].values.astype(float)
    np.savez(os.path.join(out_path, 'fluid_{0:04d}.npz'.format(current_frame)), pos=pos, vel=vel)
    if box:
        df_solid = df[df['isFluidSolid'] == 1]
        box = df_solid.loc[:, 'position_x':'position_z'].values
        np.savez(os.path.join(out_path, 'box.npz'), box=box)


def one_npz_to_csv(file_path, out_path):
    data = np.load(file_path, allow_pickle=False)
    pd.DataFrame(data).to_csv(out_path, index=False)


if __name__ == '__main__':
    file_path = r'/Users/dylan/Downloads/all_particles_2.csv'
    out_path = r'../datasets/temp'
    one_csv_to_npz(file_path, out_path, box=True)
    # one_npz_to_csv(file_path, out_path)
