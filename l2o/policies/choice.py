"""L2O that chooses either Adam or RMSProp at each iteration."""

import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .architectures import BaseCoordinateWisePolicy
from .moments import rms_momentum
from .softmax import softmax


class ChoiceOptimizer(BaseCoordinateWisePolicy):
    """L2O that chooses either Adam or RMSProp at each iteration.

    Used to show that IL with two teachers does not result in the L2O
    memorizing its teachers and simply recalling them each iteration.

    Keyword Args
    ------------
    layers : int[]
        Size of LSTM layers.
    beta_1 : float
        Momentum decay constant (table 1)
    beta_2 : float
        Variance decay constant (table 1)
    learning_rate : float
        Learning rate multiplier
    epsilon : float
        Denominator epsilon for normalization operation in case input is 0.
    hardness : float
        If hardness=0.0, uses standard softmax. Otherwise, uses gumbel-softmax
        with temperature = 1/hardness during training.
    name : str
        Name of optimizer network.
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    default_name = "ChoiceOptimizer"

    def init_layers(
            self, layers=(20, 20), beta_1=0.9, beta_2=0.999,
            learning_rate=0.001, epsilon=1e-10, hardness=0.0, **kwargs):
        """Initialize layers."""
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.hardness = hardness

        self.learning_rate = learning_rate

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]
        self.choice = Dense(2, input_shape=(layers[-1],))

    def call(self, param, inputs, states, global_state):
        """Network call override."""
        states_new = {}

        # Adam/RMSProp updates
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_rmsprop = states_new["m"] / tf.sqrt(states_new["v"] + self.epsilon)
        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = states_new["v"] / (1. - self.beta_2)
        m_tilde = m_hat / tf.sqrt(v_hat + self.epsilon)
        g_tilde = inputs / tf.sqrt(v_hat + self.epsilon)

        # Recurrent
        x = tf.concat([
            tf.reshape(m_tilde, [-1, 1]), tf.reshape(g_tilde, [-1, 1])], 1)
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])

        # Factor in softmax of Adam, RMSProp
        opt_weights = softmax(
            tf.reshape(self.choice(x), [-1, 2]),
            hardness=self.hardness, train=self.train, epsilon=self.epsilon)

        # Combine softmax
        update = self.learning_rate * (
            tf.reshape(opt_weights[:, 0], tf.shape(param)) * m_rmsprop
            + tf.reshape(opt_weights[:, 1], tf.shape(param)) * g_tilde)

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
