"""Softmax & Gumbel-softmax utilities."""

import tensorflow as tf


def softmax(weights, hardness=0.0, train=True, epsilon=1e-10):
    """Do gumbel-softmax or ordinary softmax.

    Parameters
    ----------
    weights : tf.Tensor
        Input weights to take softmax for. Dimension 0 is batch.

    Keyword Args
    ------------
    hardness : float
        Hardness value. If hardness=0.0, uses softmax.
    train : bool
        Train/test; indicates whether gumbel-softmax should be used.
    epsilon : float
        Epsilon for numerator gradient stability near 0.
    """
    dim = tf.shape(weights)[1]

    # Hard Choice
    if hardness > 0.0:
        # Train -> use gumbel-softmax approximator
        if train:
            gumbels = -tf.math.log(-tf.math.log(
                tf.random.uniform(tf.shape(weights))))
            z = tf.math.exp((weights + gumbels) * hardness + epsilon)
            return z / tf.math.reduce_sum(z, axis=1, keepdims=True)
        # Otherwise, use ordinary hard max.
        else:
            return tf.one_hot(tf.math.argmax(weights, axis=1), dim)
    # Soft Choice
    else:
        weights_exp = tf.exp(weights)
        return weights_exp / (
            tf.reduce_sum(weights_exp, axis=1, keepdims=True) + epsilon)
