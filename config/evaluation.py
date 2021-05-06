"""Evaluation problems."""


EVALUATION_PROBLEMS = {
    "mlp_train": {
        "config": {
            "layers": [20],
            "activation": "sigmoid",
        },
        "target": "mlp_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_train": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_train_dp": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128,
        "dp_args": {"clip_norm": 1.0, "noise_multiplier": 1.1}
    },
    "conv_kmnist_dp": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "kmnist",
        "epochs": 25,
        "batch_size": 128,
        "dp_args": {"clip_norm": 1.0, "noise_multiplier": 1.1}
    },
    "conv_avg": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2, [0, 3, 1]],
            "head_type": "average",
            "activation": "relu",
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_kmnist": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_train_cifar10": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "cifar10",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_cifar10": {
        "config": {
            "layers": [
                [16, 3, 1],
                [32, 3, 2],
                [32, 3, 1],
                [64, 3, 2],
                [64, 3, 1]
            ],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "cifar10",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_smallbatch": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 64
    },
    "conv_wider": {
        "config": {
            "layers": [[32, 3, 1], 2, [64, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_deeper": {
        "config": {
            "layers": [
                [16, 3, 1],
                [32, 3, 2],
                [32, 3, 1],
                [64, 3, 2],
                [64, 3, 1]
            ],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_deeper_pool": {
        "config": {
            "layers": [
                [16, 3, 1],
                [32, 3, 1],
                [32, 3, 1],
                2,
                [64, 3, 1],
                [64, 3, 1],
                2
            ],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "conv_cifar10_pool": {
        "config": {
            "layers": [
                [16, 3, 1],
                [32, 3, 1],
                [32, 3, 1],
                2,
                [64, 3, 1],
                [64, 3, 1],
                2
            ],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "cifar10",
        "epochs": 25,
        "batch_size": 128
    },
    "rastrigin2": {
        "config": {
            "n": 2, "alpha": 10,
        },
        "target": "Rastrigin",
        "steps": 1000
    },
    "rastrigin10": {
        "config": {
            "n": 10, "alpha": 10,
        },
        "target": "Rastrigin",
        "steps": 1000
    },
    "rastrigin20": {
        "config": {
            "n": 20, "alpha": 10,
        },
        "target": "Rastrigin",
        "steps": 1000
    },
    "quadratic10": {
        "config": {"n": 10},
        "target": "Quadratic",
        "steps": 1000
    },
    "quadratic20": {
        "config": {"n": 20},
        "target": "Quadratic",
        "steps": 1000
    },
    "quadratic100": {
        "config": {"n": 100},
        "target": "Quadratic",
        "steps": 1000
    },
}


def get_eval_problem(target):
    """Get evaluation config."""
    if type(target) == dict:
        return target
    try:
        return EVALUATION_PROBLEMS[target]
    except KeyError:
        raise KeyError("{} is not a valid problem.")
