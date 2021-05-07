"""Evaluate L2O."""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import l2o
from config import ArgParser, get_eval_problem
from gpu_setup import create_distribute


HELP = """
Evaluate learned optimizer.

Examples
--------
python evaluate.py --problem=conv_train --directory=weights --repeat=10

Arguments
---------
--vgpu : int >= 1
    (debug) Number of virtual GPUs to create for testing. If 1, no virtual GPUs
    are created, and a mirrored strategy is created with all physical GPUs.
--cpu : bool
    (debug) Whether to use CPU-only training.
--problem : str
    Problem to evaluate on. Can pass a comma separated list.
--directory : str
    Target directory to load from. Can pass a comma separated list.
--repeat : int
    Number of times to run evaluation.
--debug : bool
    Debug flag passed to optimizer policy.
--info : bool
    If True, prints final configuration after overrides.
--suffix : str
    Modifier to append to problem.
--strategy : str
    Strategy type to inform metadata flags. Can ignore if the default
    checkpoint is used.
--periods : int
    Periods to evaluate.
--stages : int
    Stages to evaluate. Only use if strategy=curriculum.
(all other args) : float
    Passed as overrides to strategy/policy building.
"""

if len(sys.argv) < 2:
    print(HELP)
    exit(0)

# Distribute args
args = ArgParser(sys.argv[1:])
vgpus = args.pop_get("--vgpu", default=1, dtype=int)
do_cpu = args.pop_get("--cpu", default=False, dtype=bool)
distribute = create_distribute(vgpus=vgpus, do_cpu=do_cpu)

# Core args
problems = args.pop_get("--problem", "conv_train").split(",")
targets = args.pop_get("--directory", "weights").split(",")
repeat = args.pop_get("--repeat", default=10, dtype=int)
debug = args.pop_get("--debug", default=False, dtype=bool)
show_info = args.pop_get("--info", default=False, dtype=bool)

# Suffix
suffix = args.pop_get("--suffix", "")
if suffix != "":
    suffix = "_" + suffix
if debug:
    suffix += "_dbg"

# Checkpoint specification
strategy = args.pop_get("--strategy", "repeat")
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
            for s, p in zip(stages.split(","), periods.split(","))]

# All remaining args are converted to overrides
overrides = args.to_overrides()

# Eval loop
with distribute.scope():
    for tg in targets:
        print("Strategy: {}".format(tg))
        strategy = l2o.strategy.build_from_config(
            tg, overrides=overrides, info=show_info, debug=debug)
        for m in metadata:
            print("Checkpoint: {}".format(m))
            for pr in problems:
                print("Problem: {}".format(pr))
                config = get_eval_problem(pr)
                file = pr + suffix
                strategy.evaluate(
                    metadata=m, repeat=repeat, file=file, **config)
