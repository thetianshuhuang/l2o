# Learn To Optimize
Framework for L2O extending ```tf.keras.optimizers.Optimizer```.

Todo: update with new descriptions

## Scripts
- ```resume.py```: Resume training for already-configured directory (has ```config.json``` file containing all necessary training and network parameters)
- ```train.py```: Train L2O and/or create configuration file for ```resume.py```
- ```evaluate.py```: Evaluate L2O.
- ```baseline.py```: Evaluate Adam as a baseline.

## Configuration and Hyperparameters

Overrides can be specified on the command line as ```<param_name>=<param_value>``` (i.e. ```strategy/epochs_per_period=5```)

### L2O Architecture
- ```constructor [str | l2o.optimizer.TrainableOptimizer]``` Optimizer object or object name, without the ```Optimizer``` prefix. Options: ```DM```, ```RNNProp```, ```ScaleBasic```, ```ScaleHierarchical```.

#### Hierarchical Optimizer, "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)
- ```network/param_units [int=10]```: Number of hidden units in the parameter RNN.
- ```network/tensor_units [int=5]```: Number of hidden units in the tensor RNN.
- ```network/global_units [int=5]```: Number of hidden units in the global RNN.
- ```network/init_lr [float[2]=(1e-6, 1e-2)]```: Learning rate initialization range; initial learning rates are sampled from a log-uniform distribution.
- ```network/timescales [int=5]```: Number of timescales to use for "momentum/variance at multiple timescales" technique
- ```network/epsilon [float=1e-10]```: Epsilon value used for numerical stability (i.e. to prevent divide-by-zero)
- ```network/name [str="ScaleHierarchicalOptimizer"]```: Network name.
- ```network/activation [str="tanh"]```: GRU Cell config
- ```network/recurrent_activation [str="sigmoid"]```: GRU Cell config
- ```network/use_bias [bool=True]```: GRU cell config
- ```network/kernel_initializer [str="glorot_uniform"]```: GRU Cell config
- ```network/recurrent_initializer [str="orthogonal"]```: GRU Cell config
- ```network/bias_initializer [str="zeros"]```: GRU Cell config
- ```network/unit_forget_bias [bool=True]```: GRU Cell config

#### Coordinatewise Optimizer, "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)
- ```network/layers [int[2]=(20, 20)]```: LSTM layer specifications. Length indicates depth, and value indicates number of hidden units.
- ```network/init_lr [float[2]=(1., 1.)]```: Learning rate initialization range; initial learning rates are sampled from a log-uniform distribution.
- ```network/name [str="ScaleBasicOptimizer"]```: Network name.
- ```network/activation [str="tanh"]```: LSTM Cell config
- ```network/recurrent_activation [str="sigmoid"]```: LSTM Cell config
- ```network/use_bias [bool=True]```: LSTM cell config
- ```network/kernel_initializer [str="glorot_uniform"]```: LSTM Cell config
- ```network/recurrent_initializer [str="orthogonal"]```: LSTM Cell config
- ```network/bias_initializer [str="zeros"]```: LSTM Cell config
- ```network/unit_forget_bias [bool=True]```: LSTM Cell config

#### RNNProp, "Learning Gradient Descent: Better Generalization and Longer Horizons" (Lv. et. al, 2017)
- ```network/layers [int[2]=(20, 20)]```: LSTM layer specifications. Length indicates depth, and value indicates number of hidden units.
- ```network/beta_1 [float=0.9]```: Momentum decay constant.
- ```network/beta_2 [float=0.9]```: Variance decay constant.
- ```network/alpha [float=0.1]```: Learning rate multiplier
- ```network/epsilon [float=1e-10]```: Epsilon value used for numerical stability (i.e. to prevent divide-by-zero)
- ```network/name [str="RNNPropOptimizer"]```: Network name.
- ```network/activation [str="tanh"]```: LSTM Cell config
- ```network/recurrent_activation [str="sigmoid"]```: LSTM Cell config
- ```network/use_bias [bool=True]```: LSTM cell config
- ```network/kernel_initializer [str="glorot_uniform"]```: LSTM Cell config
- ```network/recurrent_initializer [str="orthogonal"]```: LSTM Cell config
- ```network/bias_initializer [str="zeros"]```: LSTM Cell config
- ```network/unit_forget_bias [bool=True]```: LSTM Cell config

#### DMOptimizer, "Learing to learn by gradient descent by gradient descent" (Andrychowicz et. al, 2016)
- ```network/layers [int[2]=(20, 20)]```: LSTM layer specifications. Length indicates depth, and value indicates number of hidden units.
- ```network/name [str="RNNPropOptimizer"]```: Network name.
- ```network/activation [str="tanh"]```: LSTM Cell config
- ```network/recurrent_activation [str="sigmoid"]```: LSTM Cell config
- ```network/use_bias [bool=True]```: LSTM cell config
- ```network/kernel_initializer [str="glorot_uniform"]```: LSTM Cell config
- ```network/recurrent_initializer [str="orthogonal"]```: LSTM Cell config
- ```network/bias_initializer [str="zeros"]```: LSTM Cell config
- ```network/unit_forget_bias [bool=True]```: LSTM Cell config

### Training Loss
- ```training/unroll_weights [str="sum" | callable(int) -> tf.Tensor]```: Unroll weights used for loss calculation.
- ```training/teachers```: List of teachers used for imitation learning.
    - ```training/teachers/<i>/class_name [str]```: ```tf.keras.optimizers``` optimizer class name.
    - ```training/teachers/<i>/config [dict]```: Optimizer configuration keyword arguments to pass to initializer.
- ```training/p_teacher [float=0.]```: Probability of choosing imitation learning; may be overridden.
- ```training/imitation_optimizer [dict | None]```: Separate optimizer to use for imitation learning. Used to handle large differences in magnitude between meta and imitation loss.
    - ```training/imitation_optimizer/class_name [str=Adam]```: ```tf.keras.optimizers``` optimizer class name.
    - ```training/imitation_optimizer/config [dict]```: Optimizer configuration keyword arguments to pass to initializer.
- ```training/epochs [int=1]```: Number of epochs to run per problem. Applied only to batch training problems.
- ```training/depth [int=0]```: Depth, in unrolls, before parameters are reset.
- ```training/repeat [int=1]```: Number of times to repeat each problem. Applied only to full batch training problems.
- ```training/persistent [bool=False]```: Whether to keep optimizer internal state in between iterations

### Training Strategy
- ```strategy/strategy_constructor [str | l2o.train.BaseStrategy]```: Strategy object name, without the ```Strategy``` prefix. Options: ```Simple```, ```CurriculumLearning```.

#### Simple Training Strategy with Long-Tail Unroll Distribution
- ```strategy/epochs_per_period [int=10]```: Number of epochs for each period, which forms the basic unit of training
- ```strategy/num_periods [int=100]```: Number of periods to train for.
- ```strategy/unroll_distribution [float=0.05 | int | callable() -> float]```: Specify the distribution used for unrolling.
    - ```float```: ```n = np.random.geometric(unroll_distribution)```
    - ```int```: ```n = unroll_distribution```
    - ```callable() -> float```: ```n = unroll_distribution()```
- ```strategy/annealing_schedule [float=0.5 | float[] | callable(int) -> float]```: Specify the annealing schedule used to govern the probability of choosing imitation learning for each iteration.
    - ```float```: ```p = exp(-i * annealing_schedule)```
    - ```float[]```: ```p = annealing_schedule[i]```
    - ```callable(int) -> float```: ```p = annealing_schedule(i)```
- ```strategy/validation_unroll [int=50]```: Unroll length to use for validation.

#### Curriculum Learning Strategy
- ```strategy/epochs_per_period [int=10]```: Number of epochs for each period, which forms the basic unit of training
- ```strategy/schedule [dict={"base": 50, "power": 2} | callable(int) -> int | int[]]```: Curriculum learning unroll length schedule.
    - ```int[]```: ```n = schedule[i]```
    - ```callable(int) -> int```: ```n = schedule(i)```
    - ```dict```: ```base * (power ** i)```
        - ```strategy/schedule/base```: Constant coefficient multiplier
        - ```strategy/schedule/power```: Exponent base coefficient
- ```strategy/min_periods [int=10]```: Minimum number of periods per stage.
- ```strategy/max_stages [int=0]```: Maximum number of stages. If ```0```, runs until other termination conditions are reached.

### Problems
- ```problems/<i>/target```: Callable that builds the problem.
- ```problems/<i>/args```: Callable args
- ```problems/<i>/kwargs```: Callable keyword args

### Misc
- ```directory [str="weights"]```: Directory to save weights and metadata to.
- ```loss_args```: Arguments related to loss calculation
    - ```loss_args/scale_objective [bool=True]```: Scale meta learning objective value by initial value, so that the loss shows relative improvement instead of absolute loss.
    - ```loss_args/use_log_objective [bool=True]```: Take log objective value.
    - ```loss_args/obj_train_max_multiplier [float=-1]```: If the meta-loss exceeds this multiplier, iteration is terminated, and the current loss is returned. Defaults to ```-1``` (no maximum loss)
    - ```loss_args/use_numerator_epsilon [bool=True]```: Epsilon in the numerator as well as denominator (i.e. to prevent gradient of sqrt being NaN at 0)
    - ```loss_args/epsilon [float=1e-6]```: Epsilon value used for numerical stability (i.e. to prevent divide-by-zero)
- ```optimizer```: Optimizer to use for meta-optimization.
    - ```optimizer/class_name [str="Adam"]```: ```tf.keras.optimizers``` optimizer name.
    - ```optimizers/config [dic{"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999}]```: optimizer keyword args used for initialization.

## Sample Configuration

```json
{
    "problems": [
        {
            "target": "mlp_classifier",
            "args": [],
            "kwargs": {
                "layers": [20],
                "dataset": "mnist",
                "activation": "sigmoid",
                "shuffle_buffer": 16384,
                "batch_size": 8
            }
        }
    ],
    "loss_args": {
        "use_log_objective": true,
        "scale_objective": true,
        "obj_train_max_multiplier": -1,
        "use_numerator_epsilon": true,
        "epsilon": 1e-06
    },
    "optimizer": {
        "class_name": "Adam",
        "config": {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999
        }
    },
    "directory": "weights",
    "training": {
        "unroll_weights": "mean",
        "teachers": [
            {
                "class_name": "Adam",
                "config": {
                    "learning_rate": 0.001,
                    "beta_1": 0.9,
                    "beta_2": 0.999
                }
            }
        ],
        "p_teacher": 1,
        "imitation_optimizer": null,
        "strategy": "mean",
        "epochs": 1,
        "depth": 1,
        "repeat": 1,
        "persistent": false
    },
    "strategy_constructor": "Simple",
    "strategy": {
        "epochs_per_period": 10,
        "num_periods": 5,
        "unroll_distribution": 0.05,
        "annealing_schedule": 0.5,
        "validation_unroll": 50
    },
    "constructor": "DM",
    "network": {
        "layers": [20, 20],
        "name": "DMOptimizer",
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "use_bias": true,
        "kernel_initializer": "glorot_uniform",
        "recurrent_initializer": "orthogonal",
        "bias_initializer": "zeros",
        "unit_forget_bias": true
    }
}
```
