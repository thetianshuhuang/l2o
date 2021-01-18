"""Override presets."""


OVERRIDE_PRESETS = {
    "teacher_sgd": [(
        ["training", "teachers", "*"],
        {"class_name": "SGD", "config": {"learning_rate": 0.01}}
    )],
    "teacher_adam": [(
        ["training", "teachers", "*"],
        {"class_name": "Adam",
         "config": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999}}
    )],
    "teacher_rmsprop": [(
        ["training", "teachers", "*"],
        {"class_name": "RMSProp",
         "config": {"learning_rate": 0.001, "rho": 0.9}}
    )],
    "teacher_radam": [(
        ["training", "teachers", "*"],
        {
            "class_name": "RectifiedAdam",
            "config": {
                "learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999,
                "sma_threshold": 5.0, "warmup_proportion": 0.1
            }
        }
    )],
    "simple_20x25": [(
        ["strategy"],
        {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 100,
            "unroll_distribution": 20,
            "depth": 25,
            "epochs": 25,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 1,
            "validation_unroll": 50,
            "validation_depth": 25,
            "name": "SimpleStrategy"
        }
    )],
    "simple_20x50": [(
        ["strategy"],
        {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 100,
            "unroll_distribution": 20,
            "depth": 50,
            "epochs": 25,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 1,
            "validation_unroll": 50,
            "validation_depth": 25,
            "name": "SimpleStrategy"
        }
    )],
    "conv_train": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
                "activation": "relu",
                "dataset": "mnist",
                "batch_size": 128,
                "shuffle_buffer": 16384,
            }
        }]
    )],
    "conv_avg": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [[16, 3, 1], 2, [32, 5, 1], 2, [0, 3, 1]],
                "head_type": "average",
                "activation": "relu",
                "dataset": "mnist",
                "batch_size": 128,
                "shuffle_buffer": 16384
            }
        }]
    )],
    "debug": [(
        ["strategy"],
        {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 2,
            "unroll_distribution": 20,
            "depth": 10,
            "epochs": 2,
            "annealing_schedule": {"type": "constant", "value": 0.5},
            "validation_epochs": 2,
            "validation_unroll": 20,
            "name": "SimpleDebugStrategy",
        }
    )],
    "log_teachers": [
        (["training", "step_callbacks", "*"], "WhichTeacherCountCallback"),
        (["training", "stack_stats", "*"], "teacher_counts")
    ],
    "il_fast": [(
        ["strategy", "annealing_schedule"],
        {"type": "exponential", "alpha": 0.1, "base": 1.0}
    )],
    "il_slow": [(
        ["strategy", "annealing_schedule"],
        {"type": "exponential", "alpha": 0.05, "base": 1.0}
    )],
    "il_more": [(
        ["strategy", "annealing_schedule"],
        {"type": "exponential", "alpha": 0.05, "base": 10.0}
    )],
    "il_slower": [(
        ["strategy", "annealing_schedule"],
        {"type": "exponential", "alpha": 0.02, "base": 10.0}
    )],
    "il_constant": [(
        ["strategy", "annealing_schedule"],
        {"type": "constant", "value": 2.0}
    )],
    "conv_debug": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [[20, 28, 1]],
                "activation": "sigmoid",
                "dataset": "mnist",
                "batch_size": 128,
                "shuffle_buffer": 16384,
            }
        }]
    )],
    "repeat": [
        (["strategy_constructor"], "RepeatStrategy"),
        (["strategy", "max_repeat"], 4),
        (["strategy", "repeat_threshold"], 0.2),
        (["strategy", "name"], "RepeatStrategy")
    ],
    "teacher_choice": [(
        ["training", "teachers", "*"],
        {
            "class_name": "Choice",
            "config": {
                "layers": [20, 20], "beta_1": 0.9, "beta_2": 0.999,
                "learning_rate": 0.001, "epsilon": 1e-10, "hardness": 0.0,
                "activation": "tanh", "recurrent_activation": "sigmoid",
                "use_bias": True, "kernel_initializer": "glorot_uniform",
                "recurrent_initializer": "orthogonal",
                "bias_initializer": "zeros", "unit_forget_bias": True,
                "weights_file": "results/choice-20x25M-n/period_99/network"
            }
        }
    )],
    "warmup": [
        (["training", "warmup"], 5),
        (["training", "warmup_rate"], 0.01)
    ]
}


def get_preset(name):
    """Get preset override by name."""
    try:
        return OVERRIDE_PRESETS[name]
    except KeyError:
        # NOTE: We cannot use a KeyError here since KeyError has special
        # behavior which prevents it from rendering newlines correctly.
        # See https://bugs.python.org/issue2651.
        raise ValueError(
            "Invalid preset: {}. Valid presets are:\n  - ".format(name)
            + "\n  - ".join(OVERRIDE_PRESETS.keys()))
