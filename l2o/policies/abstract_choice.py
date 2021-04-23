"""L2O that chooses from a pool of optimizers at each iteration."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense

from .architectures import BaseCoordinateWisePolicy
from .moments import rms_momentum
from .softmax import softmax

from . import analytical


class AbstractChoiceOptimizer(BaseCoordinateWisePolicy):
    """L2O that chooses from a pool of optimizers at each iteration.

    Keyword Args
    ------------
    layers : int[]
        Size of LSTM layers.
    learning_rate : float
        Learning rate multiplier
    epsilon : float
        Denominator epsilon for normalization operation in case input is 0.
    hardness : float
        If hardness=0.0, uses standard softmax. Otherwise, uses gumbel-softmax
        with temperature = 1/hardness during training.
    pool : dict[]
        List of configurations for optimizers to place in the pool.
    name : str
        Name of optimizer network.
    use_meta_features : bool
        Whether to add time and tensor type features.
    lr_multiplier_scale : bool
        Maximum magnitude of log learning rate multiplier. If 0.0, the learning
        rate multiplier is not used.
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    default_name = "AbstractChoiceOptimizer"

    def init_layers(
            self, layers=(20, 20), hardness=0.0, learning_rate=0.01,
            epsilon=1e-10, pool=[],
            use_meta_features=False, lr_multiplier_scale=0.0, **kwargs):
        """Initialize layers."""
        self.choices = [
            getattr(analytical, p["class_name"] + "Optimizer")(**p["config"])
            for p in pool]

        self.hardness = hardness
        self.epsilon = epsilon

        self.use_meta_features = use_meta_features
        self.lr_multiplier_scale = lr_multiplier_scale

        self.learning_rate = learning_rate

        self.recurrent = [LSTMCell(hsize, **kwargs) for hsize in layers]
        self.choice = Dense(len(pool), input_shape=(layers[-1],))

        if self.lr_multiplier_scale > 0.0:
            self.lr_multiplier = Dense(1, input_shape=(layers[-1],))

    def call(self, param, inputs, states, global_state, training=False):
        """Network call override."""
        states_new = {}
        update, choices_new = zip(*[
            p(param, inputs, s, global_state)
            for s, p in zip(states["choices"], self.choices)
        ])
        states_new["choices"] = list(choices_new)

        # Extra features
        if self.use_meta_features:
            time_val = 1 / (1 + tf.cast(states["time"], tf.float32) / 1000)
            features = [
                tf.tile(tf.reshape(time_val, shape=(1,)), [tf.size(param)])]
            features += [
                tf.tile(
                    tf.constant(
                        tf.shape(param).shape[0] == i + 1,
                        dtype=tf.float32, shape=(1,)), [tf.size(param)])
                for i in range(3)]
            # Time
            states_new["time"] = states["time"] + 1
        else:
            features = []

        # Recurrent
        _update = [*update, *features]
        x = tf.concat([tf.reshape(x, [-1, 1]) for x in _update], 1)
        for i, layer in enumerate(self.recurrent):
            hidden_name = "rnn_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])

        # Softmax
        opt_weights = softmax(
            tf.reshape(self.choice(x), [-1, len(self.choices)]),
            hardness=self.hardness, train=training, epsilon=self.epsilon)

        if self.debug:
            states_new["_choices"] = tf.reduce_sum(opt_weights, axis=0)

        # Combine softmax
        update = self.learning_rate * sum([
            tf.reshape(opt_weights[:, i], tf.shape(param)) * u
            for i, u in enumerate(update)
        ])

        # Learning rate multiplier
        if self.lr_multiplier_scale > 0.0:
            lr_multiplier = tf.exp(
                self.lr_multiplier_scale * tf.tanh(self.lr_multiplier(x)))
            update = update * tf.reshape(lr_multiplier, tf.shape(param))
            if self.debug:
                states_new["_learning_rate"] = tf.reduce_sum(lr_multiplier)

        return update, states_new

    def get_initial_state(self, var):
        """Get initial model state as a dictionary."""
        # RNN state
        batch_size = tf.size(var)
        state = {
            "rnn_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }

        if self.use_meta_features:
            state["time"] = tf.zeros((), dtype=tf.int64)

        if (self.lr_multiplier_scale > 0.0) and self.debug:
            state["_learning_rate"] = tf.zeros((), dtype=tf.float32)

        # Child states
        state["choices"] = [p.get_initial_state(var) for p in self.choices]
        if self.debug:
            state["_choices"] = tf.zeros(len(self.choices))

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
