"""Defaults."""

from scipy.special import logit


def get_default(strategy="simple", policy="DMOptimizer"):
    """Get default configuration."""
    return dict(**BASE, **STRATEGY[strategy], **POLICY[policy])


# ------------------------------ Base Arguments ----------------------------- #

BASE = {
    "training": {
        "use_log_objective": True,
        "scale_objective": True,
        "parameter_scale_spread": 3.0,
        "loss_reduce": "reduce_max",
        "il_mode": "sum",
        "unroll_weight": "mean",
        "teachers": [],
        "obj_train_max_multiplier": -1,
        "epsilon": 1e-10,
        "step_callbacks": [],
        "pbar_values": ["meta_loss", "imitation_loss"],
        "mean_stats": ["meta_loss", "imitation_loss"],
        "stack_stats": []
    },
    "optimizer": {
        "class_name": "Adam",
        "config": {
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999
        }
    },
    "problems": [
        {
            "target": "mlp_classifier",
            "args": [],
            "kwargs": {
                "layers": [20], "dataset": "mnist", "activation": "sigmoid",
                "shuffle_buffer": 16384, "batch_size": 128
            }
        },
    ]
}

# --------------------------------- Strategy -------------------------------- #

STRATEGY = {
    "simple": {
        "strategy_constructor": "SimpleStrategy",
        "strategy": {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 50,
            "unroll_len": 20,
            "depth": 25,
            "epochs": 25,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 5,
            "validation_unroll": 20,
            "validation_depth": 50,
            "name": "SimpleStrategy",
        }
    },
    "repeat": {
        "strategy_constructor": "RepeatStrategy",
        "strategy": {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 50,
            "unroll_len": 20,
            "depth": 25,
            "epochs": 25,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 5,
            "validation_unroll": 20,
            "validation_depth": 50,
            "max_repeat": 4,
            "repeat_threshold": 0.25,
            "warmup": {"type": "constant", "value": 0},
            "warmup_rate": {"type": "constant", "value": 0.01},
            "validation_warmup": 0,
            "validation_warmup_rate": 0.01,
            "name": "RepeatStrategy",
        }
    }
}

# ------------------------ Learned Optimizer Network ------------------------ #

POLICY = {
    "scale_hierarchical": {
        "policy_constructor": "ScaleHierarchicalOptimizer",
        "policy": {
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
        "policy_constructor": "ScaleBasicOptimizer",
        "policy": {
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
        "policy_constructor": "RNNPropOptimizer",
        "policy": {
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
        "policy_constructor": "DMOptimizer",
        "policy": {
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
    },
    "choice": {
        "policy_constructor": "ChoiceOptimizer",
        "policy": {
            # RNNProp
            "layers": [20, 20],
            "beta_1": 0.9,
            "beta_2": 0.999,
            "learning_rate": 0.001,
            "epsilon": 1e-10,
            "hardness": 0.0,
            "name": "ChoiceOptimizer",
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
    "rnnprop_ext": {
        "policy_constructor": "RNNPropExtendedOptimizer",
        "policy": {
            # RNNProp
            "layers": [20, 20],
            "beta_1": 0.9,
            "beta_2": 0.999,
            "learning_rate": 0.001,
            "epsilon": 1e-10,
            "name": "RNNPropExtended",
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
    "choice_ext": {
        "policy_constructor": "ChoiceExtendedOptimizer",
        "policy": {
            # RNNProp
            "layers": [20, 20],
            "beta_1": 0.9,
            "beta_2": 0.999,
            "learning_rate": 0.001,
            "sgd_lr_multiplier": 10.,
            "epsilon": 1e-10,
            "hardness": 0.0,
            "layer_normalization": False,
            "name": "ChoiceExtendedOptimizer",
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
