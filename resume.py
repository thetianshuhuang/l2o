"""Resume Training.

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
from config import ArgParser
from gpu_setup import create_distribute


args = ArgParser(sys.argv[2:])
vgpus = int(args.pop_get("--vgpu", default=1))
distribute = create_distribute(vgpus=vgpus)

with distribute.scope():
    strategy = l2o.strategy.build_from_config(sys.argv[1])
    strategy.train()
