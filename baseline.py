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
distribute = create_distribute(vgpus=vgpus)

problem = args.pop_get("--problem", "conv_train")
target = args.pop_get("--optimizer", "adam")
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
        results.append(evaluator(tf.keras.optimizers.get(target), **kwargs))
    results = {k: np.stack([d[k] for d in results]) for k in results[0]}
    np.savez(os.path.join("baseline", target, problem), **results)
