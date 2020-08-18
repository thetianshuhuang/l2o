"""Evalute L2O."""

import sys
import l2o

folder = sys.argv[1]
stage = sys.argv[2]
period = sys.argv[3]

strategy = l2o.train.build_from_config(folder)


if stage == '*':
    stages = list(strategy.summary["stage"].unique())
else:
    stages = [int(stage)]


for stage in stages:
    if period == '*':
        periods = list(strategy._filter(stage=stage)["period"])
    else:
        periods = [int(period)]

    for period in periods:
        strategy.evaluate(stage, period)
