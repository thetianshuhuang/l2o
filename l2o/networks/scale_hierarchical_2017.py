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
        self.delta_nu = Dense(1, input_shape=(param_units,))
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
        inputs = tf.reshape(tf.reduce_mean(tf.stack(
            state["global_inputs"] for state in states), 0), [1, -1])
        global_state, _ = self.global_rnn(inputs, global_state)
        return global_state

    def _new_momentum_variance(self, grads, states):
        """Equation 1, 2, 13

        New momentum and variance
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

    def _scaled_momentum(self, states):
        """Equation 3

        Helper function for scaled gradient momentum
        """
        _m = [g_bar / tf.sqrt(lambda_) for g_bar, lambda_ in states["scaling"]]

        # m_t: [timescales, *var shape] -> [var size, timescales]
        return tf.transpose(tf.reshape(tf.stack(_m), [self.timescales, -1]))

    def _relative_log_gradient_magnitude(self, states):
        """Equation 4

        Helper function for relative log gradient magnitudes
        """
        lambdas = [lambda_ for g_bar, lambda_ in states["scaling"]]
        mean_log_magnitude = tf.reduce_mean(tf.log(tf.stack(lambdas)), axis=0)
        _gamma = [tf.log(lambda_) - mean_log_magnitude for lambda_ in lambdas]

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
        states["nu"] = self.delta_nu(states["param"]) + states["nu_bar"]
        states["nu_bar"] = (
            self.gamma * states["nu_bar"] + (1 - self.gamma) * states["nu"])

        # Direction
        # Eq 5
        # [var size, 1] -> [*var shape]
        d_theta = self.d_theta(states["param"])
        return tf.reshape(
            tf.exp(states["learning_rate"]) * d_theta * tf.size(param)
            / tf.norm(d_theta, ord=2), tf.shape(param))

    def _relative_learning_rate(self, states):
        """Equation (Unnamed), end of sec. 3.2.4

        Helper function for relative log learning rate
        """
        # nu_rel: [*var shape] -> [var size, 1]
        return tf.reshape(
            (states["learning_rate"]
             - tf.math.reduce_mean(states["learning_rate"])),
            [-1, 1])

    def call(self, param, grads, states):
        """Equation 10, 11, 13, and prerequisites

        Main call function for all except global RNN
        """

        # Eq 13 & prerequisites
        self._new_momentum_variance(grads, states)
        delta_theta = self._parameterized_change(param, states)

        # Compile x_t^(n)
        m = self._scaled_momentum(states)
        gamma = self._relative_log_gradient_magnitude(states)
        nu_rel = self._relative_learning_rate(states)

        # Param RNN
        # Eq 10
        # inputs = [var size, features]
        param_inputs = tf.concat([
            # x^n:
            m, gamma, nu_rel,
            # h_tensor: [1, hidden size] -> [var size, hidden size]
            tf.tile(states["tensor"], [tf.size(param), 1]),
            # h_global: [1, hidden size] -> [var size, hidden size]
            tf.tile(states["__global__"], [tf.size(param), 1]),
        ], 1)

        states["param"], _ = self.param_rnn(param_inputs, states["param"])

        # Compile E_tensor[x, h_param]
        # x^n: [var size, features] -> [features]
        m_tensor = tf.math.reduce_mean(m, 0)
        gamma_tensor = tf.math.reduce_mean(gamma, 0)
        nu_tensor = tf.math.reduce_mean(nu_rel, 0)
        # h_param: [var size, hidden size] -> [hidden_size]
        h_param_tensor = tf.math.reduce_mean(states["param"], 0)

        # Tensor RNN
        # Eq 11
        # inputs = [1, features]
        tensor_inputs = tf.reshape(tf.concat([
            m_tensor, gamma_tensor, nu_tensor, h_param_tensor,
            tf.reshape(states["__global__"], [-1])
        ], 0), [1, -1])

        states["tensor"], _ = self.tensor_rnn(tensor_inputs, states["tensor"])

        # Global RNN
        # Eq 12 (prepare only)
        states["global_inputs"] = tf.concat([
            m_tensor, gamma_tensor, nu_tensor, h_param_tensor,
            tf.reshape(states["tensor"], [-1])
        ], 0)

        return delta_theta, states

    def get_initial_state(self, var):

        batch_size = tf.size(var)

        def _init_lr():
            """Quick helper function to init LR and LR EMA"""
            return tf.exp(tf.random.uniform(tf.log(
                shape=tf.shape(var),
                minval=tf.log(self.init_lr[0]),
                maxval=tf.log(self.init_lr[1]))))

        return {
            "scaling": [
                [tf.zeros(tf.shape(var)), tf.zeros(tf.shape(var))]
                for s in range(self.timescales)],
            "param": self.param.get_initial_state(
                batch_size=batch_size, dtype=tf.float32),
            "tensor": self.tensor.get_initial_state(
                batch_size=1, dtype=tf.float32),
            "nu": _init_lr(),
            "nu_bar": _init_lr(),
            # m + gamma + nu + h_param + h_tensor
            "global_inputs": tf.zeros([
                self.timescales + self.timescales + 1
                + self.param_rnn.units + self.tensor_rnn.units])
        }

    def get_initial_state_global(self):
        return self.global_rnn.get_initial_state(
            batch_size=1, dtype=tf.float32)
