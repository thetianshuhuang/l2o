import l2o
import tensorflow as tf


default = dict(
    constructor=l2o.networks.DMOptimizer,
    net=l2o.train.defaults.dmoptimizer_args,
    **l2o.train.defaults.meta_learning)

default["training"]["teachers"] = [tf.keras.optimizers.Adam()]


st = l2o.train.build_curriculum(default)
st.train()
