"""Deserialize Weights."""

import tensorflow as tf


def _sum(i, n):
    """Sum loss [1 ... 1]."""
    return 1


def _mean(i, n):
    """Mean loss [1/n ... 1/n]."""
    return 1 / n


def _final(i, n):
    """Final loss [0 ... 0 1]."""
    return tf.cast(tf.equal(i, n - 1), tf.float32)


builtin_weights = {"sum": _sum, "mean": _mean, "final": _final}


def weights(x):
    """Deserialize unroll weight distribution.

    Parameters
    ----------
    x : str or Callable(int) -> tf.Tensor
        Distribution spec.

    Returns
    -------
    Callable(int) -> tf.Tensor
        Passthrough if ``x`` is Callable; otherwise, uses builtins
        ``sum``, ``mean``, or ``final``.
    """
    if callable(x):
        return x
    elif type(x) == str and x in builtin_weights:
        return builtin_weights[x]
    else:
        raise ValueError(
            "Invalid unroll weights (must be 'sum', 'mean', 'final', or "
            "callable.): {}".format(x))
