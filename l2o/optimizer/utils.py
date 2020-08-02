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
    """
    tf.nest.assert_same_structure(x, y)

    if isinstance(x, tf.Variable):
        x.assign(y)
    if isinstance(x, list) or isinstance(x, tuple):
        for _x, _y in zip(x, y):
            nested_assign(_x, _y)
    if isinstance(x, dict):
        for k in x:
            nested_assign(x[k], y[k])


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
