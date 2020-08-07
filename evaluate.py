import sys
import l2o


folder = sys.argv[1]
stage = sys.argv[2]
period = sys.argv[3]

strategy = l2o.train.build_from_config(folder)
strategy.evaluate(int(stage), int(period))
