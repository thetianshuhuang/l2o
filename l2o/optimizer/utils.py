"""Misc optimizer utilities."""

import tensorflow as tf


def wrap_variables(x, trainable=False):
    """Wrap nested structure in nested variables.

    Parameters
    ----------
    x : object
        Nested structure of tensors. Can use list, tuple, and dict.

    Keyword Args
    ------------
    trainable : bool
        Should the nested structure be trainable?

    Returns
    -------
    object
        Same nested structure as x, except with all tensors wrapped in
        variables.
    """
    if isinstance(x, tf.Tensor):
        return tf.Variable(x, trainable=trainable)
    if isinstance(x, tf.Variable):
        return x
    if isinstance(x, list):
        return [wrap_variables(y) for y in x]
    if isinstance(x, tuple):
        return tuple(wrap_variables(y) for y in x)
    if isinstance(x, dict):
        return {k: wrap_variables(v) for k, v in x.items()}


def nested_assign(x, y):
    """Assign x = y for nested structures x and y.

    Parameters
    ----------
    x : object
        Nested structure of variables to assign
    y : object
        Nested structure of variables or tensors to assign from

    Returns
    -------
    tf.python.ops.???[]
        List of ops returned by assign. Type unclear from tf source & docs.
    """
    tf.nest.assert_same_structure(x, y)

    if isinstance(x, tf.Variable):
        return [x.assign(y)]
    if isinstance(x, list) or isinstance(x, tuple):
        res = []
        for _x, _y in zip(x, y):
            res.extend(nested_assign(_x, _y))
        return res
    if isinstance(x, dict):
        res = []
        for k in x:
            res.extend(nested_assign(x[k], y[k]))
        return res
