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
strategy = args.pop_get("--strategy", "repeat")
debug = bool(args.pop_get("--debug", False))

if strategy == "repeat":
    periods = args.pop_get("--periods", None)
    if periods is None:
        metadata = [{}]
    else:
        metadata = [{"period": int(pd)} for pd in periods.split(",")]

if strategy == "curriculum":
    stages = args.pop_get("--stages", None)
    periods = args.pop_get("--periods", None)
    if stages is None:
        metadata = [{}]
    elif periods is None:
        metadata = [{"stage": int(s)} for s in stages.split(",")]
    else:
        metadata = [
            {"stage": int(s), "period": int(p)}
            for s, p in zip(stages.split(","), periods.split(","))
        ]

with distribute.scope():
    for tg in targets:
        print("Strategy: {}".format(tg))
        strategy = l2o.strategy.build_from_config(tg, info=False, debug=debug)
        for m in metadata:
            print("Checkpoint: {}".format(m))
            for pr in problems:
                print("Problem: {}".format(pr))
                config = get_eval_problem(pr)
                file = pr + "_dbg" if debug else pr
                strategy.evaluate(
                    metadata=m, repeat=repeat, file=file, **config)
