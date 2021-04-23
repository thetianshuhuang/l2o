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
        "huber_delta": -1,
        "gradient_clipping": {
            "class_name": "AdaptiveGC",
            "config": {"clip_ratio": 0.1, "epsilon": 1e-3}
        },
        "epsilon": 1e-10,
        "step_callbacks": [],
        "pbar_values": ["meta_loss", "imitation_loss"],
        "mean_stats": ["meta_loss", "imitation_loss"],
        "stack_stats": ["meta_loss", "imitation_loss"]
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
            "validation_seed": 12345,
            "num_periods": 25,
            "unroll_len": 20,
            "depth": 25,
            "epochs": 10,
            "annealing_schedule": 0.0,
            "validation_epochs": 2,
            "validation_unroll": 20,
            "validation_depth": 25,
            "warmup": 0,
            "warmup_rate": 0.01,
            "validation_warmup": 0,
            "validation_warmup_rate": 0.01,
            "name": "SimpleStrategy",
        }
    },
    "repeat": {
        "strategy_constructor": "RepeatStrategy",
        "strategy": {
            "validation_problems": None,
            "validation_seed": 12345,
            "num_periods": 4,
            "unroll_len": 100,
            "depth": {"type": "list", "values": [1, 2, 5]},
            "epochs": 10,
            "annealing_schedule": 0.0,
            "validation_epochs": 2,
            "validation_unroll": 100,
            "validation_depth": 5,
            "max_repeat": 4,
            "repeat_threshold": 0.8,
            "warmup": 0,
            "warmup_rate": 0.0,
            "validation_warmup": 0,
            "validation_warmup_rate": 0.01,
            "name": "RepeatStrategy",
        }
    },
    "curriculum": {
        "strategy_constructor": "CurriculumLearningStrategy",
        "strategy": {
            "validation_problems": None,
            "validation_seed": 12345,
            "num_stages": 4,
            "num_periods": 5,
            "num_chances": 5,
            "unroll_len": 100,
            "depth": {"type": "list", "values": [1, 2, 5, 10, 20]},
            "epochs": 10,
            "annealing_schedule": 0.0,
            "validation_epochs": 10,
            "max_repeat": 4,
            "repeat_threshold": 0.5,
            "warmup": 1,
            "warmup_rate": 0.05,
            "name": "CurriculumLearningStrategy"
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
            "warmup_lstm_update": False,
            "perturbation": None,
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
            "out_activation": "tanh",
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
    "more_choice": {
        "policy_constructor": "AbstractChoiceOptimizer",
        "policy": {
            # RNNProp
            "layers": [20, 20],
            "learning_rate": 1.0,
            "epsilon": 1e-10,
            "hardness": 0.0,
            "name": "MoreChoiceOptimizer",
            "use_meta_features": True,
            "time_scale": 1000.,
            "lr_multiplier_scale": 2.3,
            # Choices
            "pool": [
                {"class_name": "SGD", "config": {"learning_rate": 0.2}},
                {"class_name": "Momentum",
                 "config": {"learning_rate": 0.2, "beta_1": 0.9}},
                {"class_name": "RMSProp",
                 "config": {"learning_rate": 0.005, "rho": 0.9}},
                {"class_name": "Adam",
                 "config": {"learning_rate": 0.005, "beta_1": 0.9,
                            "beta_2": 0.999, "epsilon": 1e-10}},
                {"class_name": "PowerSign",
                 "config": {"learning_rate": 0.05, "beta_1": 0.9,
                            "beta_2": 0.999, "epsilon": 1e-10}},
                {"class_name": "AddSign",
                 "config": {"learning_rate": 0.05, "beta_1": 0.9,
                            "beta_2": 0.999, "epsilon": 1e-10}},
            ],
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
    "adam_lr": {
        "policy_constructor": "AdamLROptimizer",
        "policy": {
            # RNNProp
            "layers": [20, 20],
            "beta_1": 0.9,
            "beta_2": 0.999,
            "alpha": 0.1,
            "epsilon": 1e-10,
            "warmup_lstm_update": False,
            "name": "AdamLROptimizer",
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
    "rmsprop_lr": {
        "policy_constructor": "RMSPropLROptimizer",
        "policy": {
            # RNNProp
            "layers": [20, 20],
            "beta_1": 0.9,
            "beta_2": 0.999,
            "alpha": 0.1,
            "epsilon": 1e-10,
            "warmup_lstm_update": False,
            "name": "RMSPropLROptimizer",
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
    "adam": {
        "policy_constructor": "AdamOptimizer",
        "policy": {
            "learning_rate": 0.001, "beta_1": 0.9,
            "beta_2": 0.999, "epsilon": 1e-10, "trainable": True
        }
    },
    "rmsprop": {
        "policy_constructor": "RMSPropOptimizer",
        "policy": {
            "learning_rate": 0.001, "rho": 0.9, "epsilon": 1e-10,
            "trainable": True
        }
    },
    "momentum": {
        "policy_constructor": "MomentumOptimizer",
        "policy": {
            "learning_rate": 0.001, "beta_1": 0.9, "trainable": False
        }
    },
    "powersign": {
        "policy_constructor": "PowerSignOptimizer",
        "policy": {
            "learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999,
            "temperature": 1.0, "epsilon": 1e-10, "trainable": True
        }
    },
    "addsign": {
        "policy_constructor": "AddSignOptimizer",
        "policy": {
            "learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999,
            "temperature": 1.0, "epsilon": 1e-10, "trainable": True
        }
    },
    "sgd": {
        "policy_constructor": "SGDOptimizer",
        "policy": {
            "learning_rate": 0.01, "trainable": True
        }
    }
}
