"""Hand-crafted Optimizers."""

import tensorflow as tf

from .architectures import BaseCoordinateWisePolicy
from .moments import rms_momentum, rms_scaling


class AdamOptimizer(BaseCoordinateWisePolicy):
    """Adam Optimizer."""

    def init_layers(
            self, learning_rate=0.001, beta_1=0.9,
            beta_2=0.999, epsilon=1e-07):
        """Save hyperparameters (Adam has no layers)."""
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def call(self, param, inputs, states, global_state):
        """Policy call override."""
        states_new = {}
        states_new["m"], states_new["v"] = rms_momentum(
            inputs, states["m"], states["v"],
            beta_1=self.beta_1, beta_2=self.beta_2)
        m_hat = states_new["m"] / (1. - self.beta_1)
        v_hat = states_new["v"] / (1. - self.beta_2)

        update = self.learning_rate * m_hat / tf.sqrt(v_hat + self.epsilon)
        return update, states_new

    def get_initial_state(self, var):
        """Get initial optimizer state as a dictionary."""
        return {
            "m": tf.zeros(tf.shape(var)),
            "v": tf.zeros(tf.shape(var))
        }


class RMSPropOptimizer(BaseCoordinateWisePolicy):
    """RMSProp Optimizer."""

    def init_layers(self, learning_rate=0.001, rho=0.9, epsilon=1e-07):
        """Save hyperparameters."""
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def call(self, param, inputs, states, global_state):
        """Policy call override."""
        scaled, states_new = rms_scaling(
            inputs, self.rho, states, epsilon=self.epsilon)

        return scaled * self.learning_rate, states_new

    def get_initial_state(self, var):
        """Get initial optimizer state as a dictionary."""
        return tf.zeros(tf.shape(var))


class SGDOptimizer(BaseCoordinateWisePolicy):
    """SGD Optimizer."""

    def init_layers(self, learning_rate):
        """Save hyperparameters."""
        self.learning_rate = learning_rate

    def call(self, param, inputs, states, global_state):
        """Policy call override."""
        return inputs * self.learning_rate, 0.

    def get_initial_state(self, var):
        """SGD has no state."""
        return 0.
