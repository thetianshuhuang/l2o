"""Default Settings."""

from scipy.special import logit


BASE = {
    "problems": [
        {
            "target": "mlp_classifier",
            "args": [],
            "kwargs": {
                "layers": [20], "dataset": "mnist", "activation": "sigmoid",
                "shuffle_buffer": 16384, "batch_size": 64
            }
        },
    ],
    "validation_problems": None,
    "loss_args": {
        "use_log_objective": True,
        "scale_objective": True,
        "parameter_scale_spread": 3,
        "obj_train_max_multiplier": -1,
        "use_numerator_epsilon": True,
        "epsilon": 1e-10
    },
    "optimizer": {
        "class_name": "Adam",
        "config": {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999
        }
    },
}


LOSS = {
    "meta": {
        "training": {
            "unroll_weights": "mean",
            "teachers": [],
            "strategy": "mean",
            "p_teacher": 0,
            "epochs": 1,
            "depth": 0,
            "repeat": 1,
            "persistent": False,
            "imitation_threshold": 0.01,
        },
    },
    "imitation": {
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
                },
            ],
            "p_teacher": 1,
            "strategy": "mean",
            "epochs": 1,
            "depth": 0,
            "repeat": 1,
            "persistent": False,
            "imitation_threshold": 0.01,
        }
    }
}


STRATEGY = {
    "simple": {
        "strategy_constructor": "Simple",
        "strategy": {
            "epochs_per_period": 10,
            "validaton_seed": 12345,
            "num_periods": 100,
            "unroll_distribution": 0.05,
            "annealing_schedule": 0.1,
            "validation_unroll": 50
        }
    },
    "curriculum": {
        "strategy_constructor": "CurriculumLearning",
        "strategy": {
            "epochs_per_period": 10,
            "validation_seed": 12345,
            "unroll_schedule": {"coefficient": 16, "base": 2},
            "epoch_schedule": {"coefficient": 2, "base": 2},
            "annealing_schedule": 0.1,
            "annealing_floor": 0.0,
            "min_periods": 10,
            "max_stages": 3,
        },
    }
}


NETWORK = {
    "scale_hierarchical": {
        "constructor": "ScaleHierarchical",
        "network": {
            # Scale network args
            "param_units": 20,
            "tensor_units": 10,
            "global_units": 10,
            "init_lr": [1e-6, 1e-2],
            "timescales": 4,
            "epsilon": 1e-10,
            "momentum_decay_bias_init": logit(0.9),
            "variance_decay_bias_init": logit(0.999),
            "use_gradient_shortcut": True,
            "name": "ScaleHierarchicalOptimizer",
            # GRUCell args
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
        }
    },
    "scale_basic": {
        "constructor": "ScaleBasic",
        "network": {
            # Scale network args
            "layers": [20, 20],
            "init_lr": [1., 1.],
            "name": "ScaleBasicOptimizer",
            # LSTMCell Args
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
            "unit_forget_bias": True,
        }
    },
    "rnnprop": {
        "constructor": "RNNProp",
        "network": {
            # RNNProp
            "layers": [20, 20],
            "beta_1": 0.9,
            "beta_2": 0.999,
            "alpha": 0.1,
            "epsilon": 1e-10,
            "name": "RNNPropOptimizer",
            # LSTMCell Args
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
            "unit_forget_bias": True,
        }
    },
    "dmoptimizer": {
        "constructor": "DM",
        "network": {
            # DMOptimizer
            "layers": [20, 20],
            "name": "DMOptimizer",
            # LSTMCell Args
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "orthogonal",
            "bias_initializer": "zeros",
            "unit_forget_bias": True,
        }
    }
}


def get_default(loss="meta", strategy="simple", network="dmoptimizer"):
    """Get default arguments."""
    return dict(**BASE, **LOSS[loss], **STRATEGY[strategy], **NETWORK[network])
