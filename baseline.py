"""Evaluate Baseline.

Run with
```
python baseline.py directory --vgpu=1
```
"""

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
cpu = bool(args.pop_get("--cpu", default=False))
distribute = create_distribute(vgpus=vgpus, do_cpu=cpu)

problem = args.pop_get("--problem", "conv_train")

target = args.pop_get("--optimizer", "adam")
target_cfg = {
    "adam": {
        "class_name": "adam",
        "config": {"learning_rate": 0.005, "beta_1": 0.9, "beta_2": 0.999}
    },
    "rmsprop": {
        "class_name": "rmsprop",
        "config": {"learning_rate": 0.005, "rho": 0.9}
    },
    "adam_cifar": {
        "class_name": "adam",
        "config": {"learning_rate": 0.002, "beta_1": 0.9, "beta_2": 0.999}
    },
    "rmsprop_cifar": {
        "class_name": "rmsprop",
        "config": {"learning_rate": 0.002, "rho": 0.9}
    },
    "sgd": {
        "class_name": "sgd",
        "config": {"learning_rate": 0.1}
    }
}[target]

repeat = int(args.pop_get("--repeat", 10))

kwargs = get_eval_problem(problem)
if "steps" in kwargs:
    evaluator = l2o.evaluate.evaluate_function
else:
    evaluator = l2o.evaluate.evaluate_model

with distribute.scope():
    results = []
    for i in range(repeat):
        print("Evaluation Training {}/{}".format(i + 1, repeat))
        results.append(evaluator(
            tf.keras.optimizers.get(target_cfg), **kwargs))
    results = {k: np.stack([d[k] for d in results]) for k in results[0]}
    np.savez(os.path.join("baseline", target, problem), **results)
