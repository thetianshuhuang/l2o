"""Learned optimizer loading."""

import os
import json
import l2o


def load(src, name=None, enable_warmup=True):
    """Load learned optimizer.

    Parameters
    ----------
    src : str
        Packaged optimizer. Should contain a config.json, and weights.index
        checkpoint

    Keyword Args
    ------------
    name : str
        Optimizer name. If None, uses the name provided in the config file.
    enable_warmup : bool
        If True, applies warmup according to the specifications provided by
        the config file.

    Returns
    -------
    l2o.optimizer.TrainableOptimizer
        Keras-compatible optimizer.
    """
    # Load config
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)

    # Build policy (just architecture --- not optimizer yet)
    policy_constructor = l2o.deserialize.generic(
        cfg["policy_constructor"], l2o.policies, pass_cond=None,
        message='learned optimizer model', default=None)
    policy = policy_constructor(debug=False, **cfg["policy"])

    # Parse extra arguments
    if enable_warmup:
        warmup_len = cfg["settings"]["warmup"]
    else:
        warmup_len = 0

    if name is None:
        name = cfg["policy"]["name"]

    return policy.architecture(
        policy, warmup=warmup_len, warmup_rate=cfg["settings"]["warmup_rate"],
        name=name, weights_file=os.path.join(src, 'weights'))
