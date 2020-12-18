"""Keras Optimizers."""

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
    constructor = generic(
        opt["class_name"] + "Optimizer", policies,
        pass_cond=lambda x: isinstance(x, policies.BaseLearnToOptimizePolicy),
        message="gradient optimization policy", default=None)

    return constructor(**opt["config"])
