"""Coordinatewise Optimizer bundled with Code for the Scale, 2017 paper."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .moments import rms_scaling
from .network import BaseCoordinateWiseNetwork
from ..optimizer import CoordinateWiseOptimizer


class ScaleBasicOptimizer(BaseCoordinateWiseNetwork):
    """Coordinatewise version.

    Described by code accompanying
    "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)

    Keyword Args
    ------------
    layers : int[]
        Sizes of LSTM layers.
    init_lr : float | float[2]
        Learning rate initialization parameters. If ``float``, all parameter
        learning rates are initialized as a constant. If ``tuple``, the
        learning rates are initialized from a
        ``Normal(init_lr[0], init_lr[1])``.
    name : str
        Name of optimizer network.
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    architecture = CoordinateWiseOptimizer

    def __init__(
            self, layers=(20, 20), init_lr=(1., 1.),
            name="ScaleBasicOptimizer", **kwargs):

        super().__init__(name=name)

        self.init_lr = init_lr

        self.recurrent = [
            LSTMCell(hsize, name="recurrent_{}".format(i), **kwargs)
            for i, hsize in enumerate(layers)]

        self.delta = Dense(
            1, input_shape=(layers[-1],),
            name="delta")
        self.decay = Dense(
            1, input_shape=(layers[-1],), activation="sigmoid",
            name="decay")
        self.learning_rate_change = Dense(
            1, input_shape=(layers[-1],), activation="sigmoid",
            name="learning_rate_change")

    def call(self, param, inputs, states):
        states_new = {}

        # Scaling
        grad, states_new["rms"] = rms_scaling(
            inputs, states["decay"], states["rms"])

        # Recurrent
        x = tf.reshape(grad, [-1, 1])
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])

        def compute(layer):
            return tf.reshape(layer(x), tf.shape(param))

        # Update scaling hyperparameters
        states_new["decay"] = compute(self.decay)
        states_new["learning_rate"] = (
            states["learning_rate"] * 2. * compute(self.learning_rate_change))
        update = states_new["learning_rate"] * compute(self.delta)

        return update, states

    def get_initial_state(self, var):

        # RNN state
        batch_size = tf.size(var)
        rnn_state = {
            "rnn_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }

        # Learning rate random initialization
        if isinstance(self.init_lr, tuple) or isinstance(self.init_lr, list):
            init_lr = tf.exp(tf.random.uniform(
                tf.shape(var),
                np.log(self.init_lr[0]),
                np.log(self.init_lr[1])))
        else:
            init_lr = tf.constant(self.init_lr, shape=tf.shape(var))

        # State for analytical computations
        analytical_state = {
            "rms": tf.zeros(tf.shape(var)),
            "learning_rate": init_lr,
            "decay": tf.zeros(tf.shape(var))
        }

        return dict(**rnn_state, **analytical_state)
