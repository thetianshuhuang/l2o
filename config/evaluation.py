"""Evaluation problems."""


EVALUATION_PROBLEMS = {
    "train_conv": {
        "config": {
            "layers": [
                [3, 8, 1],
                [3, 16, 2],
                [3, 16, 1],
                [3, 16, 2],
                [3, 16, 1],
            ],
            "activation": "relu"
        },
        "dataset": "mnist",
        "epochs": 20,
        "batch_size": 64
    },
    "different_dataset": {
        "config": {
            "layers": [
                [3, 8, 1],
                [3, 16, 2],
                [3, 16, 1],
                [3, 16, 2],
                [3, 16, 1],
            ],
            "activation": "relu"
        },
        "dataset": "kmnist",
        "epochs": 20,
        "batch_size": 64
    },
}


def get_eval_problem(target):
    """Get evaluation config."""
    if type(target) == dict:
        return target

    try:
        return EVALUATION_PROBLEMS[name]
    except KeyError:
        raise KeyError("{} is not a valid problem.")
