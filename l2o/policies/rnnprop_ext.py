"""Extended version of RNNProp."""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .rnnprop_2016 import RNNPropOptimizer
from .moments import rms_momentum


class RNNPropExtendedOptimizer(RNNPropOptimizer):
    """Extended version of RNNProp.

    Includes an additional feature and shortcut connections to each layer.

    Keyword Args
    ------------
    layers : int[]
        Size of LSTM layers.
    beta_1 : float
        Momentum decay constant (table 1)
    beta_2 : float
        Variance decay constant (table 1)
    learning_rate : float
        Learning rate multiplier.
    epsilon : float
        Denominator epsilon for normalization operation in case input is 0.
    name : str
        Name of optimizer network.
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell

    Notes
    -----
      - Gradients are added as an input, and the inputs are provided as a
        shortcut connection to each subsequent layer.
      - The output layer has no activation.
      - The output layer has zero initialization; this seems to be critical for
        initial stability when training with IL.
    """

    default_name = "RNNPropExtended"

    def init_layers(
            self, layers=(20, 20), beta_1=0.9, beta_2=0.999,
            epsilon=1e-10, learning_rate=0.001, **kwargs):
        """Initialize Layers."""
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]

        self.delta = Dense(
            1, input_shape=(layers[-1] + 3,), activation=None,
            kernel_initializer="zeros", bias_initializer="zeros")

    def call(self, param, inputs, states, global_state):
        """Policy call override."""
        # Momentum & Variance
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)

        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = states_new["v"] / (1. - self.beta_2)
        m_tilde = m_hat / tf.sqrt(v_hat + self.epsilon)
        g_tilde = inputs / tf.sqrt(v_hat + self.epsilon)

        # Recurrent
        inputs_augmented = tf.concat([
            tf.reshape(f, [-1, 1]) for f in [inputs, m_tilde, g_tilde]], 1)
        x = inputs_augmented
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])
            x = tf.concat([x, inputs_augmented], 1)

        # Update
        update = tf.reshape(
            self.learning_rate * self.delta(x), tf.shape(param))

        return update, states_new
