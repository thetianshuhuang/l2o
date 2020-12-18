"""Main training script.

Run with
```
python train.py directory \
    --path/to/param1=param1_value
    --path/to/param2=param2_value ...
```

Optional flags:
--initialize: initialize only, but don't actually run training
--presets=preset1,preset2,...: presets to load.
"""

import sys
import tensorflow as tf

import l2o
from config import get_default, get_preset, ArgParser

# System specific settings
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.get_logger().setLevel('WARN')

# Directory always required
args = ArgParser(sys.argv[1:])
directory = args.pop_get("--directory", default="weights")

# Pick up flags first
initialize_only = args.pop_check("--initialize")

# Default params
strategy = args.pop_get("--strategy", "simple")
policy = args.pop_get("--policy", "rnnprop")
default = get_default(strategy=strategy, policy=policy)

# Build overrides
presets = args.pop_get("--presets", "")
overrides = []
for p in presets.split(','):
    overrides += get_preset(p)
overrides += args.to_overrides()

# TMP DISTRIBUTE
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512),
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
print(tf.config.experimental.list_logical_devices('GPU'))
distribute = tf.distribute.MirroredStrategy()

with distribute.scope():
    # Build strategy
    strategy = l2o.build(
        default, overrides, directory=directory, strict=True)

    # Train if not --initialize
    if not initialize_only:
        strategy.train()
