"""Override presets."""


OVERRIDE_PRESETS = {
    "sgd": [(
        ["training", "teachers", "*"],
        {"class_name": "SGD", "config": {"learning_rate": 0.05}}
    )],
    "adam": [(
        ["training", "teachers", "*"],
        {"class_name": "Adam",
         "config": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999,
                    "epsilon": 1e-10}}
    )],
    "rmsprop": [(
        ["training", "teachers", "*"],
        {"class_name": "RMSProp",
         "config": {"learning_rate": 0.001, "rho": 0.9, "epsilon": 1e-10}}
    )],
    "radam": [(
        ["training", "teachers", "*"],
        {
            "class_name": "RectifiedAdam",
            "config": {
                "learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999,
                "sma_threshold": 5.0, "warmup_proportion": 0.1
            }
        }
    )],
    "debug": [
        (["strategy", "num_periods"], 3),
        (["strategy", "unroll_len"], 20),
        (["strategy", "depth"], 5),
        (["strategy", "epochs"], 10),
        (["strategy", "validation_unroll"], 5),
        (["strategy", "validation_depth"], 10),
        (["strategy", "validation_epochs"], 1),
        (["strategy", "max_repeat"], 1),
    ],
    "conv_train": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
                "head_type": "dense",
                "activation": "relu",
                "dataset": "mnist",
                "batch_size": 128,
                "shuffle_buffer": 16384,
            }
        }]
    )],
    "conv_cifar": [(
        ["problems"],
        [{
            "target": "conv_classifier",
            "args": [],
            "kwargs": {
                "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
                "head_type": "dense",
                "activation": "relu",
                "dataset": "cifar10",
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
    "log_teachers": [
        (["training", "step_callbacks", "*"], "WhichTeacherCountCallback"),
        (["training", "stack_stats", "*"], "teacher_counts"),
    ],
    "il_constant": [
        (["strategy", "annealing_schedule"],
         {"type": "constant", "value": 0.1}),
    ],
    "il_adjusted": [
        (["strategy", "annealing_schedule"],
         {"type": "list", "values": [0.2, 0.04, 0.02, 0.01]}),
    ],
    "il_more": [
        (["strategy", "annealing_schedule"],
         {"type": "list", "values": [0.2, 0.1, 0.05, 0.02]}),
    ],
    "warmup": [
        (["strategy", "warmup"], {"type": "list", "values": [0, 5]}),
        (["strategy", "warmup_rate"], {"type": "list", "values": [0.0, 0.05]}),
        (["strategy", "validation_warmup"], 5),
        (["strategy", "validation_warmup_rate"], 0.05)
    ],
    "noscale": [
        (["training", "scale_objective"], False),
        (["strategy", "annealing_schedule", "value"], 0.25)
    ],
    "long-warmup": [
        (["strategy", "unroll_len"], 100),
        (["strategy", "depth"], 5),
        (["strategy", "validation_unroll"], 100),
        (["strategy", "validation_depth"], 5),
        (["strategy", "warmup"], {"type": "list", "values": [0, 1]}),
        (["strategy", "warmup_rate"], {"type": "list", "values": [0.0, 0.05]}),
        (["strategy", "validation_warmup"], 1),
        (["strategy", "validation_warmup_rate"], 0.05)
    ],
    "long": [
        (["strategy", "unroll_len"], 100),
        (["strategy", "depth"], 5),
        (["strategy", "validation_unroll"], 100),
        (["strategy", "validation_depth"], 5),
    ],
    "cl_fixed": [(
        ["strategy"],
        {
            "validation_problems": None,
            "validation_seed": 12345,
            "num_periods": 4,
            "unroll_len": 100,
            "depth": {"type": "list", "values": [1, 2, 5]},
            "epochs": 10,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 1,
            "validation_unroll": 100,
            "validation_depth": 5,
            "max_repeat": 4,
            "repeat_threshold": 0.9,
            "warmup": {"type": "list", "values": [0, 1]},
            "warmup_rate": {"type": "list", "values": [0.0, 0.05]},
            "validation_warmup": 1,
            "validation_warmup_rate": 0.05,
            "name": "RepeatStrategy"
        }
    )],
    "cl_unroll": [(
        ["strategy"], {
            "validation_problems": None,
            "validation_seed": 12345,
            "num_stages": 4,
            "num_periods": 2,
            "num_chances": 3,
            "unroll_len": {
                "type": "list", "values": [20, 50, 100]},
            "depth": 2,
            "epochs": 10,
            "annealing_schedule": 0.0,
            "validation_epochs": 10,
            "max_repeat": 2,
            "repeat_threshold": 0.8,
            "warmup": {"type": "list", "values": [0, 1]},
            "warmup_rate": {"type": "list", "values": [0, 0.05]},
            "name": "CurriculumLearningStrategy"
        }
    )],
    "cl_short": [
        (["strategy", "num_stages"], 3),
        (["strategy", "depth"], {"type": "list", "values": [1, 2, 5, 10]})
    ],
    "cl_long": [
        (["strategy", "num_stages"], 5),
        (["strategy", "depth"],
         {"type": "list", "values": [1, 2, 5, 10, 20, 50]})
    ],
    "half_depth": [
        (["policy", "layers"], [20])
    ],
    "fgsm": [(
        ["policy", "perturbation"], {
            "class_name": "FGSMPerturbation",
            "config": {"step_size": 0.0001}
        }
    )],
    "pgd": [(
        ["policy", "perturbation"], {
            "class_name": "PGDPerturbation",
            "config": {
                "steps": 5, "magnitude": 0.0001,
                "norm": "inf", "learning_rate": 0.1
            }
        }
    )]
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
