"""Prepare script."""

import sys
from config import ArgParser


BASE_SCRIPT = """#!/bin/sh
module load intel/18.0.2 python3/3.7.0 cuda/10.1 cudnn/7.6.5 nccl/2.5.6
python3 train.py \\
    --presets={presets} \\
    --policy={policy} \\
    --strategy={strategy} \\
    --directory=results/{policy}/{flags} \\
python3 evaluate.py \\
    --problem={problem} \\
    --directory=results/{policy}/{flags} \\
    --out={problem} \\
    --repeat=10 \\
    --periods=99
"""

BASE_RUNNER = (
    "sbatch -p gtx -N 1 -n 1 -o {policy}-{flags}.log -t 12:00:00 -A "
    "{allocation} -J {shortname} {policy}-{flags}.sh")

args = ArgParser(sys.argv[1:])

ctx = {
    "presets": args.pop_get("--presets", "conv_train"),
    "policy": args.pop_get("--policy", "rnnprop_ext"),
    "flags": args.pop_get("--flags", "test"),
    "problem": args.pop_get("--problem", "conv_train"),
    "strategy": args.pop_get("--strategy", "repeat"),
    "allocation": args.pop_get("--alloc", "Senior-Design_UT-ECE")
}

ctx["shortname"] = ctx["policy"][0].upper() + ctx["flags"]


with open("scripts/{policy}-{flags}.sh".format(**ctx), "w") as f:
    f.write(BASE_SCRIPT.format(**ctx))

print(BASE_RUNNER.format(**ctx))
