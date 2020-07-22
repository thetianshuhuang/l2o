import l2o

default = dict(
    constructor=l2o.networks.DMOptimizer,
    net=l2o.train.defaults.dmoptimizer_args,
    **l2o.train.defaults.meta_learning)

default["curriculum"]["epochs_per_period"] = 2
default["training"]["epochs"] = 2

cl = l2o.train.build_curriculum(default)
cl.train()
