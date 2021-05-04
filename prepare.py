"""Prepare script."""

import sys
import json
from config import ArgParser


BASE_SCRIPT = """#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
"""

BASE_BLOCK = """python3 train.py \\
    --presets={presets} \\
    --policy={policy} \\
    --strategy={strategy} \\
    --directory=results/{policy}/{base}/{flags}
python3 evaluate.py \\
    --problem={problem} \\
    --directory=results/{policy}/{base}/{flags} \\
    --repeat=10
"""

BLOCK_DEBUG = """
python3 evaluate.py \\
    --problem={problem} \\
    --directory=results/{policy}/{base}/{flags} \\
    --repeat=1 \\
    --debug=True
"""

BASE_RUNNER = (
    "sbatch{queue} -N 1 -n 1 -o logs/{policy}-{base}-{flags}.log -t {time}"
    "{allocation} -J {shortname}{base}{flags} "
    "scripts/{policy}-{base}-{flags}.sh")

args = ArgParser(sys.argv[1:])

flags = args.pop_get("--flags", "test").split(",")
ctx = {
    "presets": args.pop_get("--presets", "conv_train"),
    "policy": args.pop_get("--policy", "rnnprop"),
    "strategy": args.pop_get("--strategy", "repeat"),
    "allocation": args.pop_get("--alloc", "Senior-Design_UT-ECE"),
    "queue": args.pop_get("--queue", "gtx")
    "problem": args.pop_get(
        "--problem", "conv_train,conv_deeper_pool,conv_cifar10_pool"),
    "base": args.pop_get("--base", "test"),
    "time": args.pop_get("--time", "24:00:00")
}

if ctx["allocation"] != "":
    ctx["allocation"] = " -A " + ctx["allocation"]
if ctx["queue"] != "":
    ctx["queue"] = " -q " + ctx["queue"]

do_debug = bool(args.pop_get("--debug", False))
if do_debug:
    _base_block = BASE_BLOCK + BLOCK_DEBUG
else:
    _base_block = BASE_BLOCK

script = "scripts/{}-{}-{}.sh".format(ctx["policy"], ctx["base"], flags[0])
with open(script, "w") as f:
    f.write(BASE_SCRIPT + "".join(
        [_base_block.format(flags=f, **ctx) for f in flags]))


print(BASE_RUNNER.format(
    shortname=ctx["policy"][0].upper(), flags=flags[0], **ctx))
