import numpy as np
import tensorflow as tf
from tf.keras.layers import LSTMCell, Dense

from .moments import rms_scaling


class ScaleBasicOptimizer(tf.keras.Model):

    def __init__(
            self, layers, init_lr=(1., 1.), name="ScaleBasicOptimizer"):
        """RNN that operates on each coordinate independently, as specified by
        the scale paper.

        Parameters
        ----------
        """

        super(ScaleBasicOptimizer, self).__init__(name=name)

        self.init_lr = init_lr

        self.layers = [LSTMCell(hsize) for hsize in layers]

        self.delta = Dense(1, input_shape=(layers[-1],), activation="sigmoid")
        self.decay = Dense(1, input_shape=(layers[-1],))
        self.learning_rate_change = Dense(1, input_shape=(layers[-1],))

    def call(self, inputs, states):

        grad, states["rms"] = rms_scaling(
            inputs, states["decay"], states["rms"])

        x = tf.reshape(grad, [-1, 1])
        for i, layer in enumerate(self.layers):
            hidden_name = "rnn_{}".format(i)
            x, states[hidden_name] = layer(x, states[hidden_name])

        states["decay"] = self.decay(x)
        states["learning_rate"] *= 2. * self.learning_rate_change(x)
        update = tf.reshape(
            states["learning_rate"] * self.delta(x), grad.shape())

        return update, states

    def get_initial_state(self, var):

        # RNN state
        batch_size = tf.size(var)
        rnn_state = {
            "rnn_{}".format(i): layer.get_initial_state(batch_size=batch_size)
            for i, layer in enumerate(self.layers)
        }

        # Learning rate random initialization
        if type(self.init_lr) == tuple:
            init_lr = tf.exp(tf.random_uniform(
                var.get_shape(),
                np.log(self.init_lr[0]),
                np.log(self.init_lr[1])))
        else:
            init_lr = tf.constant(self.init_lr, shape=var.get_shape())

        # State for analytical computations
        analytical_state = {
            "rms": tf.ones(var.get_shape()),
            "learning_rate": init_lr,
            "decay": tf.ones(var.get_shape())
        }

        return dict(**rnn_state, **analytical_state)
