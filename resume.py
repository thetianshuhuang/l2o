"""Resume Training."""

import l2o
import sys

if len(sys.argv) != 2:
    print("resume.py takes a single argument indicating the target directory.")
    exit()

strategy = l2o.train.build_from_config(sys.argv[1])
strategy.train()
