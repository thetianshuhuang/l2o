"""Evalute L2O."""

import sys
import l2o

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

strategy = l2o.strategy.build_from_config("weights")
strategy.evaluate(
    metadata={"period": 1}, repeat=1, file="test", epochs=2, model="debug_net")

# folder = sys.argv[1]
# strategy = l2o.train.build_from_config(folder)


# if isinstance(strategy, l2o.train.CurriculumLearningStrategy):
#     stage = sys.argv[2]
#     period = sys.argv[3]

#     if stage == '*':
#         stages = list(strategy.summary["stage"].unique())
#     else:
#         stages = [int(stage)]

#     for stage in stages:
#         if period == '*':
#             periods = list(strategy._filter(stage=stage)["period"])
#         else:
#             periods = [int(period)]

#         for period in periods:
#             strategy.evaluate(stage, period)

# elif isinstance(strategy, l2o.train.SimpleStrategy):
#     period = sys.argv[2].split(",")

#     if len(period) == 1:
#         strategy.evaluate(int(period[0]), repeat=10)
#     else:
#         for p in range(int(period[0]), int(period[1])):
#             strategy.evaluate(p, repeat=10)
