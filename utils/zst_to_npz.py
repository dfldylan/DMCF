import zstandard as zstd
import collections
import msgpack
import msgpack_numpy
import shutil

msgpack_numpy.patch()
# read all data from file
from run_sample import load_data
import numpy as np
from scipy.spatial.distance import cdist

if __name__ == '__main__':
    zst_file = r'/workdir/_datasets/CCONV/dpi_dam_break/valid/0_00.msgpack.zst'
    data = load_data(zst_file)
    pos = data[000]['pos']
    # 计算所有点对之间的距离
    dist_matrix = cdist(pos, pos)

    # 提取上三角部分（排除对角线）
    triu_indices = np.triu_indices_from(dist_matrix, k=1)
    distances = dist_matrix[triu_indices]

    # 对距离和点对进行排序
    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]
    sorted_pairs = [(triu_indices[0][i], triu_indices[1][i]) for i in sorted_indices]

    print("Sorted distances between points:", sorted_distances[:10])
    # print("Corresponding point pairs:", sorted_pairs)