"""RNNProp Implementation."""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .architectures import BaseCoordinateWisePolicy
from .moments import rms_momentum


class RNNPropOptimizer(BaseCoordinateWisePolicy):
    """RNNProp algorithm.

    Described in
    "Learning Gradient Descent: Better Generalization and Longer Horizons"
    (Lv. et. al, 2017)

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
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    default_name = "RNNPropOptimizer"

    def init_layers(
            self, layers=(20, 20), beta_1=0.9, beta_2=0.9, alpha=0.1,
            epsilon=1e-10, **kwargs):
        """Initialize layers."""
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.epsilon = epsilon

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]
        self.delta = Dense(1, input_shape=(layers[-1],), activation="tanh")

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
        update = tf.reshape(self.alpha * self.delta(x), tf.shape(param))

        return update, states_new

    def get_initial_state(self, var):
        """Get initial model state as a dictionary."""
        # RNN state
        batch_size = tf.size(var)
        rnn_state = {
            "rnn_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }

        # State for analytical computations
        analytical_state = {
            "m": tf.zeros(tf.shape(var)),
            "v": tf.zeros(tf.shape(var))
        }

        return dict(**rnn_state, **analytical_state)
