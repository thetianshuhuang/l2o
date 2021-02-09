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
from config import ArgParser, get_eval_problem
from gpu_setup import create_distribute


args = ArgParser(sys.argv[1:])
vgpus = int(args.pop_get("--vgpu", default=1))
distribute = create_distribute(vgpus=vgpus)

problems = args.pop_get("--problem", "conv_train").split(",")
targets = args.pop_get("--directory", "weights").split(",")
repeat = int(args.pop_get("--repeat", 10))
periods = [int(x) for x in args.pop_get("--periods", "99").split(",")]


with distribute.scope():
    for tg in targets:
        for pd in periods:
            for pr in problems:
                strategy = l2o.strategy.build_from_config(tg)
                config = get_eval_problem(pr)
                strategy.evaluate(
                    metadata={"period": pd}, repeat=repeat, file=pr, **config)
