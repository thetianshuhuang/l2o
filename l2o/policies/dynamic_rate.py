"""Analytical Direction with Coordinatewise Dynamic Learning Rate."""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense, LayerNormalization

from .rnnprop_2016 import RNNPropOptimizer
from .moments import rms_momentum


class AdamLROptimizer(RNNPropOptimizer):
    """Optimizer restricted to the direction specified by Adam.

    The base architecture is taken from RNNProp; only the update is modified.

    Keyword Args
    ------------
    layers : int[]
        Size of LSTM layers.
    beta_1 : float
        Momentum decay constant (table 1)
    beta_2 : float
        Variance decay constant (table 1)
    alpha : float
        Learning rate multiplier (eq 7)
    epsilon : float
        Denominator epsilon for normalization operation in case input is 0.
    name : str
        Name of optimizer network.
    warmup_lstm_update : bool
        Update LSTM during warmup?
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    default_name = "AdamLR"

    def call(self, param, inputs, states, global_state):
        """Policy call override."""
        states_new = {}

        # From table 1
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = states_new["v"] / (1. - self.beta_2)

        # Eq. 5, 6
        m_tilde = m_hat / tf.sqrt(v_hat + self.epsilon)
        g_tilde = inputs / tf.sqrt(v_hat + self.epsilon)

        # Recurrent
        x = tf.concat([
            tf.reshape(m_tilde, [-1, 1]),
            tf.reshape(g_tilde, [-1, 1])
        ], 1)
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])
        # Delta
        update = (
            tf.reshape(tf.math.exp(self.delta(x)), tf.shape(param))
            * m_tilde * self.alpha)

        return update, states_new


class RMSPropLROptimizer(RNNPropOptimizer):
    """Optimizer restricted to the direction specified by RMSProp.

    The base architecture is taken from RNNProp; only the update is modified.

    Keyword Args
    ------------
    layers : int[]
        Size of LSTM layers.
    beta_1 : float
        Momentum decay constant (table 1)
    beta_2 : float
        Variance decay constant (table 1)
    alpha : float
        Learning rate multiplier (eq 7)
    epsilon : float
        Denominator epsilon for normalization operation in case input is 0.
    name : str
        Name of optimizer network.
    warmup_lstm_update : bool
        Update LSTM during warmup?
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    default_name = "AdamLR"

    def call(self, param, inputs, states, global_state):
        """Policy call override."""
        states_new = {}

        # From table 1
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = states_new["v"] / (1. - self.beta_2)

        # Eq. 5, 6
        m_tilde = m_hat / tf.sqrt(v_hat + self.epsilon)
        g_tilde = inputs / tf.sqrt(v_hat + self.epsilon)

        # Recurrent
        x = tf.concat([
            tf.reshape(m_tilde, [-1, 1]),
            tf.reshape(g_tilde, [-1, 1])
        ], 1)
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])
        # Delta
        update = (
            tf.reshape(tf.math.exp(self.delta(x)), tf.shape(param))
            * g_tilde * self.alpha)

        return update, states_new
