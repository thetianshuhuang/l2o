"""L2O that chooses either Adam or RMSProp at each iteration."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .architectures import BaseCoordinateWisePolicy
from .moments import rms_momentum
from .softmax import softmax


class ChoiceOptimizer(BaseCoordinateWisePolicy):
    """L2O that chooses either Adam or RMSProp at each iteration.

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
    warmup_lstm_update : bool
        Update LSTM during warmup?
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    default_name = "ChoiceOptimizer"

    def init_layers(
            self, layers=(20, 20), beta_1=0.9, beta_2=0.999,
            learning_rate=0.001, epsilon=1e-10, hardness=0.0,
            warmup_lstm_update=False, **kwargs):
        """Initialize layers."""
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.hardness = hardness

        self.learning_rate = learning_rate

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]
        self.choice = Dense(2, input_shape=(layers[-1],))

    def call(self, param, inputs, states, global_state, training=False):
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
            hardness=self.hardness, train=training, epsilon=self.epsilon)

        if self.debug:
            states_new["log"] = tf.reduce_sum(opt_weights, axis=0)

        # Combine softmax
        update = self.learning_rate * (
            tf.reshape(opt_weights[:, 0], tf.shape(param)) * m_rmsprop
            + tf.reshape(opt_weights[:, 1], tf.shape(param)) * g_tilde)

        return update, states_new

    def warmup_mask(self, state, new_state, in_warmup):
        """Mask state when in warmup to disable a portion of the update."""
        if self.warmup_lstm_update:
            return new_state
        else:
            rnn_state = {
                k: tf.cond(in_warmup, lambda: state[k], lambda: new_state[k])
                for k in state if k.startswith("rnn")
            }
            analytical_state = {"m": new_state["m"], "v": new_state["v"]}
            return dict(log=state["log"], **rnn_state, **analytical_state)

    def get_initial_state(self, var):
        """Get initial model state as a dictionary."""
        # RNN state
        batch_size = tf.size(var)
        state = {
            "rnn_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }

        # Debug log
        if self.debug:
            state["log"] = tf.zeros(2)

        # State for analytical computations
        state["m"] = tf.zeros(tf.shape(var))
        state["v"] = tf.zeros(tf.shape(var))

        return state

    def gather_debug(self, param, states):
        """Get debug information."""
        return states["log"]

    def debug_summarize(self, params, debug_states, debug_global):
        """Summarize debug information."""
        return {
            v.name: s.numpy() / tf.size(s).numpy()
            for v, s in zip(params, debug_states)
        }

    def aggregate_debug_data(self, data):
        """Aggregate debug data across multiple steps."""
        return {
            k: np.stack([d[k] for d in data])
            for k in data[0]
        }
