"""Main training script.

Run with
```
python train.py directory \
    --path/to/param1=param1_value
    --path/to/param2=param2_value ...
```
"""

import l2o
from config import get_default
import sys


try:
    directory = sys.argv[1]
    argv = sys.argv[2:]
except IndexError:
    print("Must provide directory argument.")

if "--resume" in argv:
    strategy = l2o.train.build_from_config(directory)
    strategy.train()

else:
    default = l2o.train.get_default(
        loss="imitation", strategy="curriculum", network="rnnprop")
    trainer = l2o.train.build_argv(
        default, directory=sys.argv[1], argv=sys.argv[2:])
    trainer.train()
