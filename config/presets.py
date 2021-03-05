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
        (["strategy", "depth"], 20),
        (["strategy", "epochs"], 2),
        (["strategy", "validation_unroll"], 20),
        (["strategy", "validation_depth"], 2),
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
    "il_constant": [
        (["strategy", "annealing_schedule"],
         {"type": "constant", "value": 0.01}),
        (["strategy", "num_periods"], 50),
        (["training", "step_callbacks", "*"], "WhichTeacherCountCallback"),
        (["training", "stack_stats", "*"], "teacher_counts"),
    ],
    "warmup": [
        (["strategy", "warmup"], {"type": "list", "values": [0, 5]}),
        (["strategy", "warmup_rate"], {"type": "list", "values": [0.0, 0.05]}),
        (["strategy", "validation_warmup"], 5),
        (["strategy", "validation_warmup_rate"], 0.05)
    ],
    "huber": [
        (["training", "huber_delta"], 0.01)
    ],
    "clip": [(
        ["training", "gradient_clipping"],
        {"class_name": "AdaptiveGC",
         "config": {"clip_ratio": 0.1, "epsilon": 1e-3}}
    )],
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
