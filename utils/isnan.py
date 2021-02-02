"""Inspect checkpoint for NaN."""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tgt = sys.argv[1]
reader = tf.train.load_checkpoint(tgt)

keys = [k for k in reader.get_variable_to_shape_map().keys() if k[0] != '_']

print("Checking checkpoint {} for NaN:".format(tgt))
for k in keys:
    tensor = reader.get_tensor(k)
    if tensor.dtype == "float32" or tensor.dtype == "float16":
        is_nan = tf.reduce_sum(
            tf.cast(tf.math.is_nan(tensor), tf.int32)).numpy()
        if is_nan > 0:
            print("{}: {} / {}".format(k, is_nan, tf.size(tensor).numpy()))
