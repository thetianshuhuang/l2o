"""Override presets."""


OVERRIDE_PRESETS = {
    "teacher_adam": [(
        ["training", "teachers", "*"],
        {"class_name": "adam",
         "config": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999}}
    )],
    "teacher_rmsprop": [(
        ["training", "teachers", "*"],
        {"class_name": "rmsprop",
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
            "epochs": 50,
            "repeat": 5,
            "annealing_schedule": 0.5,
            "validation_repeat": 1,
            "validation_unroll": 200,
            "name": "SimpleStrategy"
        }
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
    )]
}


def get_preset(name):
    """Get preset override by name."""
    return OVERRIDE_PRESETS.get(name, [])
