import tensorflow as tf
from tensorflow.keras.layers import GRUCell, Dense

from .moments import rms_momentum


class ScaleHierarchicalOptimizer(tf.keras.Model):
    """Scale network; inherits tf.keras.Model

    Keyword Args
    ------------
    param_units : int
        Number of hidden units for parameter RNN.
    tensor_units : int
        Number of hidden units for tensor RNN.
    global_units : int
        Number of hidden units for global RNN.
    init_lr : float[2]
        Learning rate initialization range. Actual learning rate values are
        IID exp(unif(log(init_lr))).
    timescales : int
        Number of timescales to compute momentum for.
    name : str
        Name of optimizer network
    **kwargs : dict
        Passed onto tf.keras.layers.GRUCell
    """

    def __init__(
            self, param_units=10, tensor_units=5, global_units=5,
            init_lr=(1e-6, 1e-2), timescales=1,
            name="ScaleHierarchicalOptimizer", **kwargs):

        super().__init__(name=name)

        self.timescales = timescales
        self.init_lr = init_lr

        # Parameter, Tensor, & Global RNNs (may have different size)
        self.param_rnn = GRUCell(param_units, **kwargs)
        self.tensor_rnn = GRUCell(tensor_units, **kwargs)
        self.global_rnn = GRUCell(global_units, **kwargs)

        # Parameter change
        self.d_theta = Dense(1, input_shape=(param_units,))
        # Learning rate change
        self.delta_nu = Dense(
            1, input_shape=(param_units,), activation="sigmoid")
        # Momentum decay rate
        self.beta_g = Dense(
            1, input_shape=(param_units,), activation="sigmoid")
        # Variance/scale decay rate
        self.beta_lambda = Dense(
            1, input_shape=(param_units,), activation="sigmoid")

        # Gamma parameter
        self.gamma = tf.Variable(tf.zeros(()), trainable=True, name="gamma")

    def call_global(self, states, global_state):
        """Equation 12

        Global RNN. Inputs are prepared (except for final mean) in ``call``.
        """
        # [1, units] -> [num tensors, 1, units] -> [1, units]
        inputs = tf.reduce_mean(tf.stack(
            state["tensor"] for state in states), 0)
        global_state, _ = self.global_rnn(inputs, global_state)
        return global_state

    def _new_momentum_variance(self, grads, states):
        """Equation 1, 2, 3, 13

        Helper function for scaled momentum update
        """
        # Base decay
        # Eq 13
        # [var size, 1] -> [*var shape]
        shape = tf.shape(grads)
        beta_g = tf.reshape(self.beta_g(states["param"]), shape)
        beta_lambda = tf.reshape(self.beta_lambda(states["param"]), shape)

        # New momentum, variance
        # Eq 1, 2
        for s, (g_bar, lambda_) in enumerate(states["scaling"]):
            states["scaling"][s] = rms_momentum(
                grads, g_bar, lambda_,
                beta_1=beta_g ** (0.5 ** s), beta_2=beta_lambda ** (0.5**s))

        # Scaled momentum
        _m = [g_bar / tf.sqrt(lambda_) for g_bar, lambda_ in states["scaling"]]

        # m_t: [timescales, *var shape] -> [var size, timescales]
        return tf.transpose(tf.reshape(tf.stack(_m), [self.timescales, -1]))

    def _relative_log_gradient_magnitude(self, states):
        """Equation 4

        Helper function for relative log gradient magnitudes
        """
        lambdas = [lambda_ for g_bar, lambda_ in states["scaling"]]
        mean_lambda = tf.reduce_mean(tf.math.log(tf.stack(lambdas)), axis=0)
        _gamma = [tf.math.log(lambda_) - mean_lambda for lambda_ in lambdas]

        # gamma_t: [timescales, *var shape] -> [var size, timescales]
        return tf.transpose(
            tf.reshape(tf.stack(_gamma), [self.timescales, -1]))

    def _parameterized_change(self, param, states):
        """Equation 5, 7, 8

        Helper function for parameter change explicitly parameterized into
        direction and learning rate
        """
        # New learning rate
        # Eq 7, 8
        eta = self.delta_nu(states["param"]) + states["eta_bar"]
        states["eta_bar"] = (
            self.gamma * states["eta_bar"] + (1 - self.gamma) * eta)

        # Relative log learning rate
        # Eq Unnamed, end of sec 3.2.4
        eta_rel = tf.reshape(eta - tf.math.reduce_mean(eta), [-1, 1])

        # Direction
        # Eq 5
        # [var size, 1] -> [*var shape]
        d_theta = self.d_theta(states["param"])
        delta_theta = tf.reshape(
            tf.exp(eta) * d_theta * tf.cast(tf.size(param), tf.float32)
            / tf.norm(d_theta, ord=2), tf.shape(param))

        return delta_theta, eta_rel

    def call(self, param, grads, states, global_state):
        """Equation 10, 11, 13, and prerequisites

        Main call function for all except global RNN
        """

        # Prerequisites
        # Eq 1, 2, 3, 13
        m = self._new_momentum_variance(grads, states)
        # Eq 5, 7, 8
        delta_theta, eta_rel = self._parameterized_change(param, states)
        # Eq 4
        gamma = self._relative_log_gradient_magnitude(states)

        # Param RNN
        # inputs = [var size, features]
        param_inputs = tf.concat([
            # x^n:
            m, gamma, eta_rel,
            # h_tensor: [1, hidden size] -> [var size, hidden size]
            tf.tile(states["tensor"], [tf.size(param), 1]),
            # h_global: [1, hidden size] -> [var size, hidden size]
            tf.tile(global_state, [tf.size(param), 1]),
        ], 1)

        # RNN Update
        # Eq 10
        states["param"], _ = self.param_rnn(param_inputs, states["param"])
        # Eq 11
        tensor_inputs = tf.concat([
            tf.math.reduce_mean(states["param"], 0, keepdims=True),
            global_state
        ], 1)
        states["tensor"], _ = self.tensor_rnn(tensor_inputs, states["tensor"])

        return delta_theta, states

    def get_initial_state(self, var):

        batch_size = tf.size(var)

        return {
            "scaling": [
                [tf.zeros(tf.shape(var)), tf.zeros(tf.shape(var))]
                for s in range(self.timescales)],
            "param": self.param_rnn.get_initial_state(
                batch_size=batch_size, dtype=tf.float32),
            "tensor": self.tensor_rnn.get_initial_state(
                batch_size=1, dtype=tf.float32),
            "eta_bar": tf.exp(tf.random.uniform(
                shape=tf.shape(var),
                minval=tf.math.log(self.init_lr[0]),
                maxval=tf.math.log(self.init_lr[1]))),
        }

    def get_initial_state_global(self):
        return self.global_rnn.get_initial_state(
            batch_size=1, dtype=tf.float32)
