"""Evaluate L2O.

Run with
```
python evaluate.py directory --vgpu=1
```
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import l2o
from config import create_distribute, ArgParser, get_eval_problem


args = ArgParser(sys.argv[1:])
vgpus = int(args.pop_get("--vgpu", default=1))
distribute = create_distribute(vgpus=vgpus)

problem = args.pop_get("--problem", "conv_train")
target = args.pop_get("--directory", "weights")
output = args.pop_get("--out", "eval")

with distribute.scope():
    strategy = l2o.strategy.build_from_config(directory)
    config = get_eval_problem("debug")
    strategy.evaluate(
        metadata={"period": 99}, repeat=10, file=output, **config)
