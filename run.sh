python train.py problems/0/kwargs/batch_size=32 strategy/epochs_per_period=5 strategy/min_periods=20 training/epochs=5 strategy/annealing_schedule=0.1
python train.py problems/0/kwargs/batch_size=64 strategy/epochs_per_period=5 strategy/min_periods=20 training/epochs=5 strategy/annealing_schedule=0.1
python train.py problems/0/kwargs/batch_size=64 strategy/epochs_per_period=5 strategy/min_periods=20 'strategy/epoch_schedule={"coefficient": 5, "base": 2}' strategy/annealing_schedule=0.1
python train.py problems/0/kwargs/batch_size=64 strategy/epochs_per_period=5 strategy/min_periods=20 'strategy/epoch_schedule={"coefficient": 1, "base": 2}' strategy/annealing_schedule=0.1 training/repeat=5