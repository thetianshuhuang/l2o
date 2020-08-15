"""Main training script.

Run with
```
python train.py directory \
    --path/to/param1=param1_value
    --path/to/param2=param2_value ...
```

Optional flags:
--initialize: initialize only, but don't actually run training
--resume: resume training config in directory. Ignores all other args.
--presets=preset1,preset2,...: presets to load.
"""

import l2o
from config import get_default, ArgParser, get_preset
import sys


# Directory always required
args = ArgParser(sys.argv[1:])
directory = argparse.pop_get("directory", default="weights")

# Pick up flags first
initialize_only = argparse.pop_check("--initialize")

# Resume -> ignore all other args
if argparse.pop_check("--resume", kwargs):
    strategy = l2o.train.build_from_config(directory)
    strategy.train()
    exit(0)

# Execute presets
presets = argparse.pop_get("--presets", None)
overrides = []
for p in presets.split(','):
    overrides += get_preset(p)

# Build overrides
overrides += args.to_overrides()

# Build strategy
default = l2o.train.get_default(
    loss="imitation", strategy="curriculum", network="rnnprop")
trainer = l2o.train.build(
    default, overrides, directory=directory, saved_config=True, strict=True)

# Train if not --initialize
if not initialize_only:
    trainer.train()
