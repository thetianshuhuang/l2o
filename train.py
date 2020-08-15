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

import l2o
from config import get_default, ArgParser, get_preset
import sys

# Directory always required
args = ArgParser(sys.argv[1:])
directory = args.pop_get("directory", default="weights")

# Pick up flags first
initialize_only = args.pop_check("--initialize")

# Build overrides
presets = args.pop_get("--presets", "")
overrides = []
for p in presets.split(','):
    overrides += get_preset(p)
overrides += args.to_overrides()

# Build strategy
default = get_default(
    loss="imitation", strategy="curriculum", network="rnnprop")
trainer = l2o.train.build(
    default, overrides, directory=directory, saved_config=True, strict=True)

# Train if not --initialize
if not initialize_only:
    trainer.train()
