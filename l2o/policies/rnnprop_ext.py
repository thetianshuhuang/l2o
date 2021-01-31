"""Extended version of RNNProp."""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .rnnprop_2016 import RNNPropOptimizer
from .moments import rms_momentum


class RNNPropExtendedOptimizer(RNNPropOptimizer):
    """Extended version of RNNProp.

    Includes an additional feature and shortcut connections to each layer.
    """

    default_name = "RNNPropExtended"

    def init_layers(
            self, layers=(20, 20), beta_1=0.9, beta_2=0.999,
            epsilon=1e-10, **kwargs):
        """Initialize Layers."""
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]

        self.learning_rate = Dense(
            1, input_shape=(layers[-1] + 3,), activation=None)
        self.delta = Dense(
            1, input_shape=(layers[-1] + 3,), activation=None)

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
            tf.exp(self.learning_rate(x)) * self.delta(x), tf.shape(param))

        return update, states_new