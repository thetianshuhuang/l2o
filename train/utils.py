"""Misc training utilities."""

import tensorflow as tf
import numpy as np


# NOTE: Currently Unused
def reset_optimizer(opt):
    """Reset tensorflow optimizer.

    Parameters
    ----------
    opt : tf.keras.optimizers.Optimizer
        Optimizer to reset; replaces all variables with 0. Note that non-zero
        initializations are not respected since the variables do not store
        their initial value.
    """
    for var in opt.variables():
        var.assign(tf.zeros_like(var))


def make_seeds(seed, n):
    """Make random integer seeds.

    Parameters
    ----------
    seed : int or None
        Integer seed to use. If None, passes None along.
    n : int
        Size of array to generate.
    """
    if seed is None:
        return [None] * n
    else:
        rng = np.random.default_rng(seed)
        return rng.integers(-2147483648, 2147483647, size=n, dtype=np.int32)


# NOTE: Currently Unused
def regen_optimizer_vars(optimizers, trainable_variables):
    """Force optimizers to generate hidden state variables.

    As of 2.3.0-rc2, I believe f.keras.optimizers.Optimizer has a
    compatibility issue with get_concrete_function. Using
    get_concrete_function triggers two traces, and sometimes causes issues
    on the second retrace with the optimizer trying to create variables.
    Therefore, this method forcibly generates hidden variables outside of
    the @tf.function loss functions to avoid this bug.
    """
    if len(optimizers) > 0:
        for opt, var_set in zip(optimizers, trainable_variables):
            opt._create_all_weights(var_set)
