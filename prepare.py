"""Prepare script."""

import sys
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

BASE_RUNNER = (
    "sbatch -p gtx -N 1 -n 1 -o logs/{policy}-{base}-{flags}.log -t 24:00:00 "
    "-A {allocation} -J {shortname}{base}{flags} "
    "scripts/{policy}-{base}-{flags}.sh")

args = ArgParser(sys.argv[1:])

flags = args.pop_get("--flags", "test").split(",")
ctx = {
    "presets": args.pop_get("--presets", "conv_train"),
    "policy": args.pop_get("--policy", "rnnprop"),
    "strategy": args.pop_get("--strategy", "repeat"),
    "allocation": args.pop_get("--alloc", "Senior-Design_UT-ECE"),
    "problem": args.pop_get("--problem", "conv_train"),
    "base": args.pop_get("--base", "test")
}


with open(
        "scripts/{}-{}-{}.sh".format(ctx["policy"], base, flags[0]), "w") as f:
    f.write(BASE_SCRIPT + "".join(
        [BASE_BLOCK.format(flags=f, **ctx) for f in flags]))


print(BASE_RUNNER.format(
    shortname=ctx["policy"][0].upper(), flags=flags[0], **ctx))
