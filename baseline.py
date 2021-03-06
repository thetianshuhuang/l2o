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
from config import create_distribute, ArgParser, get_eval_problem


args = ArgParser(sys.argv[1:])
vgpus = int(args.pop_get("--vgpu", default=1))
distribute = create_distribute(vgpus=vgpus)

problem = args.pop_get("--problem", "conv_train")
target = args.pop_get("--optimizer", "adam")
output = args.pop_get("--out", "eval")
repeat = int(args.pop_get("--repeat", 10))

with distribute.scope():
    results = []
    for i in range(repeat):
        print("Evaluation Training {}/{}".format(i + 1, repeat))
        results.append(l2o.evaluate.evaluate(
            tf.keras.optimizers.get(target), **get_eval_problem(problem)))
    results = {k: np.stack([d[k] for d in results]) for k in results[0]}
    np.savez(output, **results)
