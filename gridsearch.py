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
# policy_name = args.pop_get("--optimizer", "adam")

problem = {
    "config": {
        "layers": [
            [16, 3, 1],
            [32, 3, 1],
            [32, 3, 1],
            2,
            [64, 3, 1],
            [64, 3, 1],
            2
        ],
        "activation": "relu"
    },
    "target": "conv_classifier",
    "dataset": "cifar10",
    "epochs": 25,
    "batch_size": 128
}

learning_rates = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
policy_names = ["adam", "rmsprop", "sgd", "momentum", "addsign", "powersign"]

for policy_name in policy_names:
    for lr in learning_rates:
        if policy_name == "adam":
            policy = l2o.policies.AdamOptimizer(
                learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        elif policy_name == "rmsprop":
            policy = l2o.policies.RMSPropOptimizer(
                learning_rate=lr, rho=0.9)
        elif policy_name == "sgd":
            policy = l2o.policies.SGDOptimizer(learning_rate=lr)
        elif policy_name == "momentum":
            policy = l2o.policies.MomentumOptimizer(
                learning_rate=lr, beta_1=0.9)
        elif policy_name == "addsign":
            policy = l2o.policies.AddSignOptimizer(
                learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10)
        elif policy_name == "powersign":
            policy = l2o.policies.PowerSignOptimizer(
                learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-10)

        opt = policy.architecture(policy)

        with distribute.scope():
            results = l2o.evaluate.evaluate_model(
                opt, **get_eval_problem(problem))
            np.savez(
                "gridsearch/conv_cifar10_pool/{}/{}".format(policy_name, lr),
                **results)
