"""Main training script.

Run with
```
python train.py
    --directory=weights
    --path/to/param1=param1_value
    --path/to/param2=param2_value ...
```

Optional flags:
--initialize: initialize only, but don't actually run training
--presets=preset1,preset2,...: presets to load.
"""

import os
import sys

from config import get_default, get_preset, ArgParser

args = ArgParser(sys.argv[1:])
gpu_number = args.pop_get("--target_gpu", None)

# Finally ready to import tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if gpu_number is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
import tensorflow as tf
import l2o
from gpu_setup import create_distribute

# Directory
directory = args.pop_get("--directory", default="weights")

# Distribute
vgpus = int(args.pop_get("--vgpu", default=1))
distribute = create_distribute(vgpus=vgpus)

# Pick up flags first
initialize_only = args.pop_check("--initialize")

# Default params
strategy = args.pop_get("--strategy", "repeat")
policy = args.pop_get("--policy", "rnnprop")
default = get_default(strategy=strategy, policy=policy)

# Build overrides
presets = args.pop_get("--presets", "")
overrides = []
if presets != "":
    for p in presets.split(','):
        overrides += get_preset(p)
overrides += args.to_overrides()

with distribute.scope():
    # Build strategy
    strategy = l2o.build(
        default, overrides, directory=directory, strict=True)

    # Train if not --initialize
    if not initialize_only:
        strategy.train()
