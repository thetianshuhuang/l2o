"""L2O that chooses either Adam or RMSProp at each iteration."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .architectures import BaseCoordinateWisePolicy
from .moments import rms_momentum
from .softmax import softmax


class ChoiceLargeOptimizer(BaseCoordinateWisePolicy):
    """L2O that chooses from the ``large" pool at each iteration.

    Keyword Args
    ------------
    layers : int[]
        Size of LSTM layers.
    beta_1 : float
        Momentum decay constant (table 1)
    beta_2 : float
        Variance decay constant (table 1)
    learning_rates : dict(str -> float)
        Learning rate multiplier for each policy.
    epsilon : float
        Denominator epsilon for normalization operation in case input is 0.
    time_scale : float
        Denominator scale for time feature; the input feature is
        ```min(t / time_scale, 1)```.
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
            learning_rate=None, epsilon=1e-10, time_scale=2000.,
            warmup_lstm_update=False, **kwargs):
        """Initialize layers."""
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.time_scale = time_scale
        self.warmup_lstm_update = warmup_lstm_update

        if learning_rate is None:
            learning_rate = {
                "sgd": 0.2, "momentum": 0.5,
                "rmsprop": 0.005, "adam": 0.005,
                "powersign": 0.1, "addsign": 0.1
            }
        self.learning_rate = learning_rate

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]
        self.choice = Dense(6, input_shape=(layers[-1],))

    def call(self, param, inputs, states, global_state, training=False):
        """Network call override."""
        states_new = {}

        # Feature updates
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        states_new["time"] = states["time"] + 1
        _time = tf.cast(states["time"], tf.float32) / self.time_scale

        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = tf.sqrt(states_new["v"] / (1. - self.beta_2) + self.epsilon)
        m_tilde = m_hat / v_hat
        g_tilde = inputs / v_hat

        # Input: g, m, m/sqrt(v), g/sqrt(v), f(t)
        x = tf.concat([
            tf.reshape(inputs, [-1, 1]),
            tf.reshape(states_new["m"], [-1, 1]),
            tf.reshape(g_tilde, [-1, 1]),
            tf.reshape(m_tilde, [-1, 1]),
            tf.tile(tf.reshape(1 / (1 + _time), (1, 1)), [tf.size(param), 1])
        ], 1)

        # Recurrent
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])

        # Choices
        pool = {
            "sgd": inputs,
            "momentum": states_new["m"],
            "rmsprop": g_tilde,
            "adam": m_tilde,
            "powersign": tf.exp(
                tf.math.tanh(g_tilde) * tf.math.tanh(m_tilde)) * inputs,
            "addsign": (
                1 + tf.math.tanh(g_tilde) * tf.math.tanh(m_tilde)) * inputs
        }

        # Factor in softmax of Adam, RMSProp
        opt_weights = softmax(
            tf.reshape(self.choice(x), [-1, 6]),
            train=training, epsilon=self.epsilon)
        if self.debug:
            states_new["_choices"] = tf.reduce_sum(opt_weights, axis=0)

        # Combine softmax
        update = sum([
            update * self.learning_rate[key] * tf.reshape(
                opt_weights[:, i], tf.shape(param))
            for i, (key, update) in enumerate(pool.items())
        ])

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
            analytical_state = {
                k: v for k, v in new_state.items() if k not in rnn_state}
            return dict(**rnn_state, **analytical_state)

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
            state["_choices"] = tf.zeros(2)

        # State for analytical computations
        state["m"] = tf.zeros(tf.shape(var))
        state["v"] = tf.zeros(tf.shape(var))
        state["time"] = tf.zeros((), dtype=tf.int64)

        return state

    def debug_summarize(self, params, debug_states, debug_global):
        """Summarize debug information."""
        return {
            k + "_" + p.name: v / tf.cast(tf.size(p), tf.float32)
            for p, s in zip(params, debug_states)
            for k, v in s.items()
        }

    def aggregate_debug_data(self, data):
        """Aggregate debug data across multiple steps."""
        return {
            k: np.stack([d[k] for d in data])
            for k in data[0]
        }
