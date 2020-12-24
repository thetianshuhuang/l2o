"""Evaluation problems."""


EVALUATION_PROBLEMS = {
    "debug": {
        "config": {},
        "target": "debug_net",
        "dataset": "mnist",
        "epochs": 2,
        "batch_size": 32
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
    "different_dataset": {
        "config": {
            "layers": [[16, 3, 1], 2, [32, 5, 1], 2],
            "activation": "relu"
        },
        "target": "conv_classifier",
        "dataset": "mnist",
        "epochs": 25,
        "batch_size": 128
    },
    "smaller_batch": {
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
}


def get_eval_problem(target):
    """Get evaluation config."""
    if type(target) == dict:
        return target
    try:
        return EVALUATION_PROBLEMS[target]
    except KeyError:
        raise KeyError("{} is not a valid problem.")
