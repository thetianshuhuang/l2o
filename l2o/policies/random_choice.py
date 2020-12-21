"""Baseline optimzier that randomly selects Adam or RMSProp."""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .architectures import BaseCoordinateWisePolicy
from .moments import rms_momentum


class RandomChoiceOptimizer(BaseCoordinateWisePolicy):
    """Baseline optimzier that randomly selects Adam or RMSProp.

    Keyword Args
    ------------
    beta_1 : float
        Momentum decay constant (table 1)
    beta_2 : float
        Variance decay constant (table 1)
    learning_rate : float
        Learning rate multiplier
    name : str
        Name of optimizer network.
    adam_weight : float
        Probability of choosing Adam. Otherwise, chooses RMSProp.
    """

    default_name = "RandomChoiceOptimizer"

    def init_layers(
            self, beta_1=0.9, beta_2=0.999, learning_rate=0.001,
            adam_weight=0.5):
        """Initialize layers."""
        self.learning_rate = learning_rate
        self.adam_weight = 0.5

    def call(self, param, inputs, states, global_state):
        """Network call override."""
        states_new = {}

        # Adam/RMSProp updates
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_rmsprop = states_new["m"] / tf.sqrt(states_new["v"] + self.epsilon)
        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = states_new["v"] / (1. - self.beta_2)
        m_tilde = m_hat / tf.sqrt(v_hat + self.epsilon)
        g_tilde = inputs / tf.sqrt(v_hat + self.epsilon)

        rvs = tf.random.uniform(tf.shape(param), minval=0.0, maxval=0.0)
        adam_weight = tf.cast(tf.math.less(rvs, self.adam_weight), tf.float32)

        # Combine softmax
        update = self.learning_rate * (
            tf.reshape(1 - adam_weight, tf.shape(param)) * m_rmsprop
            + tf.reshape(adam_weight, tf.shape(param)) * g_tilde)

        return update, states_new

    def get_initial_state(self, var):
        """Get initial model state as a dictionary."""
        return {
            "m": tf.zeros(tf.shape(var)),
            "v": tf.zeros(tf.shape(var))
        }
