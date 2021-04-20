"""Hyperparameter grid search."""

import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import l2o
from config import ArgParser, get_eval_problem
from gpu_setup import create_distribute


args = ArgParser(sys.argv[1:])
vgpus = int(args.pop_get("--vgpu", default=1))
distribute = create_distribute(vgpus=vgpus)
policy_name = args.pop_get("--optimizer", "adam")

problem = {
    "config": {
        "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
        "activation": "relu"
    },
    "target": "conv_classifier",
    "dataset": "mnist",
    "epochs": 5,
    "batch_size": 128
}


for lr in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
    if policy_name == "adam":
        policy = l2o.policies.AdamOptimizer(
            learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    elif policy_name == "rmsprop":
        policy = l2o.policies.RMSPropOptimizer(
            learning_rate=lr, rho=0.9)
    elif policy_name == "sgd":
        policy = l2o.policies.SGDOptimizer(learning_rate=lr)
    elif policy_name == "momentum":
        policy = l2o.policies.MomentumOptimizer(learning_rate=lr, beta_1=0.9)

    opt = policy.architecture(policy)

    with distribute.scope():
        results = l2o.evaluate.evaluate(opt, **get_eval_problem(problem))
        np.savez("gridsearch/{}/{}".format(policy, lr), **results)
