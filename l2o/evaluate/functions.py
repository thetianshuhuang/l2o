"""Functions for plain GD evaluation."""

import tensorflow as tf
import math


class Rastrigin:
    """Rastrigin function."""

    def __init__(self, n=10, alpha=10):

        self.n = n
        self.alpha = alpha

        self.A = tf.random.normal([n, n])
        self.b = tf.random.normal([n])
        self.c = tf.random.normal([n])

        self.x = tf.Variable(tf.zeros([n]))
        self.trainable_variables = [self.x]

    def loss(self):
        """Compute function loss."""
        return (
            tf.reduce_sum(tf.square(tf.linalg.matvec(self.A, self.x) - self.b))
            - self.alpha * tf.reduce_sum(
                self.c * tf.math.cos(2 * math.pi * self.x))
            + self.alpha * self.n)


class Quadratic:
    """Quadratic function."""

    def __init__(self, n=10):
        self.n = n

        self.A = tf.random.normal([n, n])
        self.b = tf.random.normal([n])

        self.x = tf.Variable(tf.zeros([n]))
        self.trainable_variables = [self.x]

    def loss(self):
        """Compute function loss."""
        return tf.reduce_sum(
            tf.square(tf.linalg.matvec(self.A, self.x) - self.b))
