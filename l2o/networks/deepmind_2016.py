"""Deepmind, 2016 optimizer implementation"""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .network import BaseCoordinateWiseNetwork
from ..optimizer import CoordinateWiseOptimizer


class DMOptimizer(BaseCoordinateWiseNetwork):
    """DMOptimizer algorithm as described in
    "Learing to learn by gradient descent by gradient descent"
    (Andrychowicz et. al, 2016)

    Keyword Args
    ------------
    layers : int[]
        Sizes of LSTM layers.
    name : str
        Name of optimizer network.
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    architecture = CoordinateWiseOptimizer

    def __init__(self, layers=(20, 20), name="DMOptimizer", **kwargs):

        super().__init__(name=name)

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]
        self.delta = Dense(1, input_shape=(layers[-1],))

    def call(self, param, inputs, states):
        states_new = {}

        x = tf.reshape(inputs, [-1, 1])
        for i, layer in enumerate(self.recurrent):
            hidden_name = "lstm_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])
        x = self.delta(x)

        return tf.reshape(x, param.shape), states_new

    def get_initial_state(self, var):
        batch_size = tf.size(var)
        return {
            "lstm_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }
