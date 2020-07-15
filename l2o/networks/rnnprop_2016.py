"""RNNProp Implementation"""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .moments import rms_momentum


class RNNPropOptimizer(tf.keras.Model):

    def __init__(
            self, layers=(20, 20), beta_1=0.9, beta_2=0.9, alpha=0.1,
            epsilon=1e-10, name="RNNPropOptimizer", **kwargs):
        """RNNProp algorithm as described by Better Generalization.

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

        References
        ----------
        K. Lv, S. Jiang, J. Li. "Learning Gradient Descent: Better
        Generalization and Longer Horizons," ICML 34, 2017.
        """

        super().__init__(name=name)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.epsilon = epsilon

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]
        self.delta = Dense(1, input_shape=(layers[-1],), activation="tanh")

    def call(self, param, inputs, states):

        # From table 1
        states["m"], states["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_hat = states["m"] / (1. - self.beta_1)
        v_hat = states["v"] / (1. - self.beta_2)

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
            x, states[hidden_name] = layer(x, states[hidden_name])
        # Delta
        update = tf.reshape(self.alpha * self.delta(x), tf.shape(param))

        return update, states

    def get_initial_state(self, var):

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
