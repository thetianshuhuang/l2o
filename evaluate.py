"""Evalute L2O."""

import sys
import l2o

folder = sys.argv[1]
strategy = l2o.train.build_from_config(folder)


if isinstance(strategy, l2o.train.CurriculumLearningStrategy):
    stage = sys.argv[2]
    period = sys.argv[3]

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

elif isinstance(strategy, l2o.train.SimpleStrategy):
    period = sys.argv[2].split(",")

    if len(period) == 1:
        strategy.evaluate(int(period[0]))
    else:
        strategy.evaluate(range(int(period[0]), int(period[1])))
