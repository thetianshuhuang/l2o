import l2o

cl = l2o.train.build_curriculum(dict(
    constructor=l2o.networks.DMOptimizer,
    net=l2o.train.defaults.dmoptimizer_args,
    **l2o.train.defaults.meta_learning))
cl.train()
