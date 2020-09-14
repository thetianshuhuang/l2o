"""Override presets."""


OVERRIDE_PRESETS = {
    "imitation_optimizer": [
        (
            ["optimizer"],
            {"class_name": "rmsprop",
             "config": {"learning_rate": 0.001, "rho": 0.9}}
        ), (
            ["training", "imitation_optimizer"],
            {"class_name": "rmsprop",
             "config": {"learning_rate": 0.001, "rho": 0.9}}
        )
    ],
    "rmsprop_teacher": [
        (
            ["training", "teachers", "*"],
            {"class_name": "rmsprop",
             "config": {"learning_rate": 0.001, "rho": 0.9}}
        )
    ],
    "simple_comparison": [
        (["training", "epochs"], 50),
        (["training", "repeat"], 5),
        (["strategy", "unroll_distribution"], 200),
        (["strategy", "epochs_per_period"], 1),
        (["strategy", "annealing_schedule"], 0.05),
        (["strategy", "num_periods"], 100)
    ],
    "radam_teacher": [
        (
            ["training", "teachers", "*"], {
                "class_name": "RectifiedAdam",
                "config": {
                    "learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999,
                    "sma_threshold": 5.0, "warmup_proportion": 0.1
                }
            }
        )
    ],
    "nadam_teacher": [
        (
            ["training", "teachers", "*"], {
                "class_name": "Nadam",
                "config": {
                    "learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999
                }
            }
        )
    ]
}


def get_preset(name):
    """Get preset override by name."""
    return OVERRIDE_PRESETS.get(name, [])
