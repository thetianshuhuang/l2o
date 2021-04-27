"""Override presets."""


OVERRIDE_PRESETS = {

    # ----------------------------------------------------------------------- #
    #                                 Teachers                                #
    # ----------------------------------------------------------------------- #

    "sgd": [(
        ["training", "teachers", "*"],
        {"class_name": "SGD", "config": {"learning_rate": 0.2}}
    )],
    "momentum": [(
        ["training", "teachers", "*"],
        {"class_name": "Momentum",
         "config": {"learning_rate": 0.5, "beta_1": 0.9}},
    )],
    "adam": [(
        ["training", "teachers", "*"],
        {"class_name": "Adam",
         "config": {"learning_rate": 0.005, "beta_1": 0.9, "beta_2": 0.999,
                    "epsilon": 1e-10}}
    )],
    "rmsprop": [(
        ["training", "teachers", "*"],
        {"class_name": "RMSProp",
         "config": {"learning_rate": 0.005, "rho": 0.9, "epsilon": 1e-10}}
    )],
    "powersign": [(
        ["training", "teachers", "*"],
        {"class_name": "PowerSign",
         "config": {"learning_rate": 0.1, "beta_1": 0.9,
                    "beta_2": 0.999, "epsilon": 1e-10}},
    )],
    "addsign": [(
        ["training", "teachers", "*"],
        {"class_name": "AddSign",
         "config": {"learning_rate": 0.1, "beta_1": 0.9,
                    "beta_2": 0.999, "epsilon": 1e-10}},
    )],
    "choice": [(
        ["training", "teachers", "*"],
        {
            "class_name": "__load__",
            "directory": "results/choice/base/1",
            "checkpoint": "period_3.0"
        }
    )],
    "more_choice": [(
        ["training", "teachers", "*"],
        {
            "class_name": "__load__",
            "directory": "results/more-choice/base/1",
            "checkpoint": "period_4.0"
        }
    )],

    # ----------------------------------------------------------------------- #
    #                                 Problems                                #
    # ----------------------------------------------------------------------- #

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

    # ----------------------------------------------------------------------- #
    #                                 Training                                #
    # ----------------------------------------------------------------------- #

    "debug": [
        (["strategy", "num_periods"], 3),
        (["strategy", "unroll_len"], 20),
        (["strategy", "depth"], 2),
        (["strategy", "epochs"], 25),
        (["strategy", "validation_unroll"], 5),
        (["strategy", "validation_depth"], 10),
        (["strategy", "validation_epochs"], 1),
        (["strategy", "max_repeat"], 1),
    ],
    "log_teachers": [
        (["training", "step_callbacks", "*"], "WhichTeacherCountCallback"),
        (["training", "stack_stats", "*"], "teacher_counts"),
    ],
    "il_adjusted": [
        (["strategy", "annealing_schedule"],
         {"type": "list", "values": [0.2, 0.04, 0.02, 0.01]}),
    ],
    "il_more": [
        (["strategy", "annealing_schedule"],
         {"type": "list", "values": [0.2, 0.1, 0.05, 0.02]}),
        (["training", "step_callbacks", "*"], "WhichTeacherCountCallback"),
        (["training", "stack_stats", "*"], "teacher_counts"),
    ],
    "cl_fixed": [(
        ["strategy"],
        {
            "validation_problems": None,
            "validation_seed": 12345,
            "num_periods": 5,
            "unroll_len": 100,
            "depth": {"type": "list", "values": [1, 2, 5]},
            "epochs": 10,
            "annealing_schedule": {"type": "constant", "value": 0.0},
            "validation_epochs": 1,
            "validation_unroll": 100,
            "validation_depth": 10,
            "max_repeat": 4,
            "repeat_threshold": 0.9,
            "warmup": {"type": "list", "values": [0, 1]},
            "warmup_rate": {"type": "list", "values": [0.0, 0.05]},
            "validation_warmup": 1,
            "validation_warmup_rate": 0.05,
            "name": "RepeatStrategy"
        }
    )],
    "half_depth": [
        (["policy", "layers"], [20])
    ],
    "hard": [(["policy", "hardness"], 10.0)],
    "noscale": [(["training", "parameter_scale_spread"], 0.0)]

    # ----------------------------------------------------------------------- #
    #                              Perturbations                              #
    # ----------------------------------------------------------------------- #

    "fgsm": [(
        ["policy", "perturbation"], {
            "class_name": "FGSMPerturbation",
            "config": {"step_size": 0.001}
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
    )],
    "cgd": [(
        ["policy", "perturbation"], {
            "class_name": "CGDPerturbation",
            "config": {"steps": 3, "magnitude": 0.005}
        }
    )],
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
