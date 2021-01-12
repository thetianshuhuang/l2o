"""Load Saved Policy."""

import os
import json
import pprint

from l2o import optimizer


def load(policy, directory="weights", name=None):
    """Load trained policy from directory.

    Parameters
    ----------
    policy : l2o.policies.BaseLearnToOptimizePolicy
        Policy to load weights for.

    Keyword Args
    ------------
    directory : str
        Target directory. Must contain:
          - <config.json>: policy configuration.
          - <network.index>, <network.data*>: saved tf.train.Checkpoint.
    name : str
        Optimizer name. If None, uses the default_name registered by the
        policy.

    Returns
    -------
    tf.keras.optimizers.Optimizer
        Keras-compatible optimizer.
    """

    try:
        with open(os.path.join(directory, "config.json"), 'r') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise Exception("Directory must contain a 'config.json' file.")

    print("Loading {}:{}".format(policy.default_name, directory))
    print("Config:")
    pprint.pprint(cfg)

    network = policy(**cfg)
    optimizer = policy.architecture(
        network, weights_file=os.path.join(directory, "network"))

    return optimizer
