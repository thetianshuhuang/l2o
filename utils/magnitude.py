"""Show parameters."""

import os
import sys
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tgt = sys.argv[1]
reader = tf.train.load_checkpoint(tgt)


for k, v in reader.get_variable_to_shape_map().items():
    if (
            re.match("network/.*/.ATTRIBUTES/VARIABLE_VALUE", k)
            and "OPTIMIZER_SLOT" not in k):
        ms_val = tf.sqrt(tf.reduce_mean(tf.square(reader.get_tensor(k))))
        m_val = tf.reduce_mean(tf.math.abs(reader.get_tensor(k)))
        print(
            k.replace("/.ATTRIBUTES/VARIABLE_VALUE", ""),
            ms_val.numpy(), m_val.numpy())
