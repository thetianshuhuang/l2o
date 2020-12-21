"""Evaluate L2O.

Run with
```
python resume.py directory --vgpu=1
```
"""

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import l2o
from config import create_distribute, ArgParser


args = ArgParser(sys.argv[2:])
vgpus = int(args.pop_get("--vgpu", default=1))
distribute = create_distribute(vgpus=vgpus)

with distribute.scope():
    strategy = l2o.strategy.build_from_config(sys.argv[1])
    config = {
        "config": {},
        "target": "debug_net",
        "dataset": "mnist",
        "epochs": 2,
        "batch_size": 16
    }
    strategy.evaluate(
        metadata={"period": 1}, repeat=1, file="test", **config)
