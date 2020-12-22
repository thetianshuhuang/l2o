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
    "teacher_nadam": [(
        ["training", "teachers", "*"],
        {"class_name": "Nadam",
         "config": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999}}
    )],
    "simple_comparison": [(
        ["strategy"],
        {
            "validation_problems": None,
            "epochs_per_period": 1,
            "validation_seed": 12345,
            "num_periods": 100,
            "unroll_distribution": 200,
            "epochs": 200,
            "repeat": 5,
            "annealing_schedule": 0.5,
            "validation_repeat": 1,
            "validation_unroll": 200,
            "name": "SimpleStrategy"
        }
    )],
    "train_conv": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [
                    [3, 8, 1],
                    [3, 16, 2],
                    [3, 16, 1],
                    [3, 16, 2],
                    [3, 16, 1],
                ],
                "activation": "relu",
                "dataset": "mnist",
                "batch_size": 64,
                "shuffle_buffer": 16384,
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
            "epochs": 1,
            "repeat": 1,
            "annealing_schedule": 0.5,
            "validation_repeat": 1,
            "validation_unroll": 20,
            "name": "SimpleDebugStrategy",
        }
    )],
    "log_teachers": [
        (["training", "step_callbacks", "*"], "WhichTeacherCountCallback"),
        (["training", "stack_stats", "*"], "teacher_counts")
    ]
}


def get_preset(name):
    """Get preset override by name."""
    try:
        return OVERRIDE_PRESETS[name]
    except KeyError:
        raise KeyError(
            "Invalid preset: {}.\nValid presets are:\n  - ".format(name)
            + "\n  - ".join(OVERRIDE_PRESETS.keys()))
