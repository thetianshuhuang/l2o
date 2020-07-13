import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .moments import rms_scaling


class ScaleBasicOptimizer(tf.keras.Model):

    def __init__(
            self, layers=(20, 20), init_lr=(1., 1.),
            name="ScaleBasicOptimizer"):
        """RNN that operates on each coordinate independently, as specified by
        the scale paper.

        Parameters
        ----------
        """

        super(ScaleBasicOptimizer, self).__init__(name=name)

        self.init_lr = init_lr

        self.recurrent = [LSTMCell(hsize) for hsize in layers]

        self.delta = Dense(1, input_shape=(layers[-1],), activation="sigmoid")
        self.decay = Dense(1, input_shape=(layers[-1],))
        self.learning_rate_change = Dense(1, input_shape=(layers[-1],))

    def call(self, param, inputs, states):

        grad, states["rms"] = rms_scaling(
            inputs, states["decay"], states["rms"])

        x = tf.reshape(grad, [-1, 1])
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states[hidden_name] = layer(x, states[hidden_name])

        states["decay"] = self.decay(x)
        states["learning_rate"] *= 2. * self.learning_rate_change(x)
        update = tf.reshape(
            states["learning_rate"] * self.delta(x), param.shape)

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
        if type(self.init_lr) == tuple:
            init_lr = tf.exp(tf.random.uniform(
                tf.shape(var),
                np.log(self.init_lr[0]),
                np.log(self.init_lr[1])))
        else:
            init_lr = tf.constant(self.init_lr, shape=tf.shape(var))

        # State for analytical computations
        analytical_state = {
            "rms": tf.ones(tf.shape(var)),
            "learning_rate": init_lr,
            "decay": tf.ones(tf.shape(var))
        }

        return dict(**rnn_state, **analytical_state)
