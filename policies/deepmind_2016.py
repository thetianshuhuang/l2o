"""Deepmind, 2016 optimizer implementation."""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .architectures import BaseCoordinateWisePolicy


class DMOptimizer(BaseCoordinateWisePolicy):
    """DMOptimizer algorithm.

    Described in
    "Learing to learn by gradient descent by gradient descent"
    (Andrychowicz et. al, 2016)

    Keyword Args
    ------------
    learning_rate : float
        Initial learning rate multiplier.
    layers : int[]
        Sizes of LSTM layers.
    name : str
        Name of optimizer network.
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    default_name = "DMOptimizer"

    def init_layers(self, learning_rate=0.1, layers=(20, 20), **kwargs):
        """Initialize layers."""
        self.learning_rate = learning_rate
        self.recurrent = [
            LSTMCell(hsize, name="recurrent_{}".format(i), **kwargs)
            for i, hsize in enumerate(layers)]
        self.delta = Dense(1, input_shape=(layers[-1],), name="delta")

    def call(self, param, inputs, states, global_state, training=False):
        """Network call override."""
        states_new = {}

        x = tf.reshape(inputs, [-1, 1])
        for i, layer in enumerate(self.recurrent):
            hidden_name = "lstm_{}".format(i)
            x, states_new[hidden_name] = layer(
                x, states[hidden_name], training=training)
        x = self.delta(x, training=training)

        return self.learning_rate * tf.reshape(x, param.shape), states_new

    def get_initial_state(self, var):
        """Get initial model state as a dictionary."""
        batch_size = tf.size(var)
        return {
            "lstm_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }
