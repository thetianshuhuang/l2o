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
            and not "OPTIMIZER_SLOT" in k):
        print(k, v)
