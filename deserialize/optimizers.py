"""Keras Optimizers."""

import os
import json
import tensorflow as tf

from l2o import policies
from .generic import generic


def optimizer(opt):
    """Helper function to get optimizer using tf.keras.optimizers.get.

    Also includes optimizers in tensorflow_addons.optimizers if available.
    """
    # Mainline keras optimizers
    try:
        return tf.keras.optimizers.get(opt)
    # Not in tf.keras.optimizers -> try tensorflow_addons
    except ValueError as e:
        # In tensorflow_addons -> replicate tf.keras.optimizers.get behavior
        try:
            from tensorflow_addons import optimizers as tfa_opts
            if isinstance(opt, str):
                return getattr(tfa_opts, opt)()
            elif isinstance(opt, dict):
                return getattr(tfa_opts, opt['class_name'])(**opt['config'])
        # tensorflow_addons not available -> raise original error.
        except ModuleNotFoundError:
            print(
                "Warning: tensorflow_addons is not available. Only Keras "
                "Optimizers were searched for a match.")
            raise(e)


def policy(opt):
    """Helper function to get optimizer policy from l2o.policies."""
    # Directory specification
    if opt["class_name"] == "__load__":
        with open(os.path.join(opt["directory"], "config.json"), 'r') as f:
            cfg = json.load(f)

        constructor = generic(cfg["policy_constructor"], policies)
        src = os.path.join(opt["directory"], "checkpoint", opt["checkpoint"])
        return constructor(weights_file=src, **cfg["policy"])

    # Explicit specification
    else:
        constructor = generic(
            opt["class_name"] + "Optimizer", policies,
            pass_cond=lambda x: isinstance(
                x, policies.BaseLearnToOptimizePolicy),
            message="gradient optimization policy", default=None)

        return constructor(**opt["config"])
