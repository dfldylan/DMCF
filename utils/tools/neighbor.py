import tensorflow as tf


def reduce_subarrays_sum_multi(values, row_splits):
    row_splits = tf.cast(row_splits, tf.int32)
    segment_ids = tf.ragged.row_splits_to_segment_ids(row_splits)
    sum = tf.math.unsorted_segment_sum(values, segment_ids, num_segments=tf.shape(row_splits)[0] - 1)
    return sum


def combine_nns(fluid_nns, solid_nns):
    # 拆分两个邻居搜索的结果
    fluid_index, fluid_row_splits, fluid_distance = fluid_nns
    solid_index, solid_row_splits, solid_distance = solid_nns
    fluid_segment_ids = tf.ragged.row_splits_to_segment_ids(fluid_row_splits)
    solid_segment_ids = tf.ragged.row_splits_to_segment_ids(solid_row_splits)

    # 合并neighbors_index，neighbors_distance，以及segment_ids
    combined_index = tf.concat([fluid_index, solid_index + (tf.shape(fluid_row_splits)[0] - 1)], axis=0)
    combined_distance = tf.concat([fluid_distance, solid_distance], axis=0)
    combined_ids = tf.concat([fluid_segment_ids, solid_segment_ids], axis=0)

    # 根据combined_ids对neighbors_index，neighbors_distance进行排序
    rank = tf.argsort(combined_ids)
    combined_index = tf.gather(combined_index, rank)
    combined_distance = tf.gather(combined_distance, rank)
    combined_ids = tf.gather(combined_ids, rank)

    combined_row_splits = tf.ragged.segment_ids_to_row_splits(combined_ids)
    return combined_index, combined_row_splits, combined_distance


def neighbors_mask(neighbors, mask):
    neighbors_index, neighbors_row_splits, neighbors_distance = neighbors

    # 使用新的mask来过滤neighbors_index和neighbors_distance
    neighbors_counts = neighbors_row_splits[1:] - neighbors_row_splits[:-1]
    expanded_mask = tf.repeat(mask, neighbors_counts, axis=0)
    neighbors_index_filtered = tf.boolean_mask(neighbors_index, expanded_mask)
    neighbors_distance_filtered = tf.boolean_mask(neighbors_distance, expanded_mask)

    # 更新neighbors_row_splits
    row_splits_diff_filtered = tf.boolean_mask(neighbors_counts, mask)
    neighbors_row_splits_filtered = tf.concat([[0], tf.cumsum(row_splits_diff_filtered)], axis=0)

    # 更新neighbors，使其包含过滤后的tensor
    neighbors_filtered = (neighbors_index_filtered, neighbors_row_splits_filtered, neighbors_distance_filtered)
    return neighbors_filtered
