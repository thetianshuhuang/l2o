"""Gradient Clipping Algorithms."""

import tensorflow as tf


class SimpleGC:
    """Simple Gradient Clipping.

    Parameters
    ----------
    clip_value : float
        Clipping threshold. If <= 0, no gradient clipping is performed.
    """

    def __init__(self, clip_value=10.0):
        self.clip_value = clip_value

    def clip(self, params, grads):
        """Clip Gradients."""
        if self.clip_value > 0:
            return [
                tf.clip_by_value(g, -self.clip_value, self.clip_value)
                for g in grads]
        else:
            return grads


class AdaptiveGC:
    """Adaptive Gradient Clipping.

    Described by
    "High-Performance Large-Scale Image Recognition Without Normalization"
    (Brock et. al, 2021)

    Parameters
    ----------
    clip_ratio : float
        Clipping threshold lambda.
    epsilon : float
        Minimum normalization magnitude.
    """

    def __init__(self, clip_ratio=0.1, epsilon=1e-3):
        self.clip_ratio = clip_ratio
        self.epsilon = epsilon

    def _clip(self, p, g):
        # ||W||*_F
        weight_norm = tf.math.maximum(
            tf.math.sqrt(tf.math.reduce_sum(tf.math.square(p))), self.epsilon)
        # ||G||_F
        grad_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(g)))
        # Eq 3
        return tf.cond(
            grad_norm > weight_norm * self.clip_ratio,
            lambda: weight_norm / grad_norm * self.clip_ratio * g,
            lambda: g)

    def clip(self, params, grads):
        """Clip Gradients."""
        return [self._clip(p, g) for p, g in zip(params, grads)]
