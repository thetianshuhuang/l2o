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
    "simple_20": [(
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
    "simple_50": [(
        ["strategy"],
        {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 100,
            "unroll_distribution": 50,
            "depth": 25,
            "epochs": 10,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 1,
            "validation_unroll": 50,
            "validation_depth": 25,
            "name": "SimpleStrategy"
        }
    )],
    "simple_100": [(
        ["strategy"],
        {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 100,
            "unroll_distribution": 100,
            "depth": 25,
            "epochs": 5,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 1,
            "validation_unroll": 50,
            "validation_depth": 25,
            "name": "SimpleStrategy"
        }
    )],
    "simple_250": [(
        ["strategy"],
        {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 100,
            "unroll_distribution": 250,
            "depth": 25,
            "epochs": 2,
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
    "conv_train_bigger": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [[16, 7, 1], 2, [32, 5, 1], 2],
                "activation": "relu",
                "dataset": "mnist",
                "batch_size": 128,
                "shuffle_buffer": 16384,
            }
        }]
    )],
    "conv_train_wider": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [[32, 3, 1], 2, [32, 5, 1], 2],
                "activation": "relu",
                "dataset": "mnist",
                "batch_size": 128,
                "shuffle_buffer": 16384,
            }
        }]
    )],
    "mlp_train_deeper": [(
        ["problems"],
        [{
            "target": "mlp_classifier",
            "args": [],
            "kwargs": {
                "layers": [20, 20], "dataset": "mnist",
                "activation": "sigmoid", "shuffle_buffer": 18384,
                "batch_size": 128
            }
        }]
    )],
    "leaky_relu": [(
        ["problems", 0, "kwargs", "activation"],
        {"class_name": "relu", "config": {"alpha": 0.01}}
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
    "metaopt_sgd": [(
        ["optimizer"],
        {"class_name": "sgd", "config": {"learning_rate": 0.01}}
    )],
    "il_fast": [(
        ["strategy", "annealing_schedule"],
        {"type": "exponential", "alpha": 0.1}
    )],
    "il_slow": [(
        ["strategy", "annealing_schedule"],
        {"type": "exponential", "alpha": 0.04}
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
    )]
}


def get_preset(name):
    """Get preset override by name."""
    try:
        return OVERRIDE_PRESETS[name]
    except KeyError:
        raise KeyError(
            "Invalid preset: {}.\nValid presets are:\n  - ".format(name)
            + "\n  - ".join(OVERRIDE_PRESETS.keys()))
