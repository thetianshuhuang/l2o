"""Hierarchical Optimizer described by the Scale, 2017 paper."""
import tensorflow as tf
from tensorflow.keras.layers import GRUCell, Dense
from scipy.special import logit

from .network import BaseHierarchicalNetwork
from .moments import rms_momentum


class ScaleHierarchicalOptimizer(BaseHierarchicalNetwork):
    """Hierarchical optimizer.

    Described in
    "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)

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
    epsilon : float
        Denominator epsilon for normalization operation in case input is 0.
    momentum_decay_bias_init : float
        Constant initializer for EMA momentum decay rate logit beta_g. Should
        correspond to beta_1 in an Adam teacher.
    variance_decay_bias_init : float
        Constant initializer for EMA variance decay rate logit beta_lambda.
        Should correspond to beta_2 in an Adam teacher.
    use_gradient_shortcut : bool
        Use shortcut connection adding linear transformation of momentum at
        various timescales to direction output?
    name : str
        Name of optimizer network
    **kwargs : dict
        Passed onto tf.keras.layers.GRUCell
    """

    def __init__(
            self, param_units=10, tensor_units=5, global_units=5,
            init_lr=(1e-6, 1e-2), timescales=1, epsilon=1e-10,
            momentum_decay_bias_init=logit(0.9),
            variance_decay_bias_init=logit(0.999),
            use_gradient_shortcut=True,
            name="ScaleHierarchicalOptimizer", **kwargs):

        super().__init__(name=name)

        assert(init_lr[0] > 0 and init_lr[1] > 0 and epsilon > 0)
        self.timescales = timescales
        self.init_lr = init_lr
        self.epsilon = epsilon

        # Parameter, Tensor, & Global RNNs (may have different size)
        self.param_rnn = GRUCell(param_units, name="param_rnn", **kwargs)
        self.tensor_rnn = GRUCell(tensor_units, name="tensor_rnn", **kwargs)
        self.global_rnn = GRUCell(global_units, name="global_rnn", **kwargs)

        # Parameter change
        self.d_theta = Dense(1, input_shape=(param_units,), name="d_theta")
        # Learning rate change
        self.delta_nu = Dense(1, input_shape=(param_units,), name="delta_nu")
        # Momentum decay rate
        self.beta_g = Dense(
            1, input_shape=(param_units,), kernel_initializer="zeros",
            bias_initializer=tf.constant_initializer(
                value=momentum_decay_bias_init),
            activation="sigmoid", name="beta_g")
        # Variance/scale decay rate
        self.beta_lambda = Dense(
            1, input_shape=(param_units,), kernel_initializer="zeros",
            bias_initializer=tf.constant_initializer(
                value=variance_decay_bias_init),
            activation="sigmoid", name="beta_lambda")
        # Momentum shortcut
        if use_gradient_shortcut:
            self.gradient_shortcut = Dense(
                1, input_shape=(timescales,), use_bias=False,
                name="gradient_shortcut")

        # Gamma parameter
        # Stored as a logit - the actual gamma used will be sigmoid(gamma)
        self.gamma = tf.Variable(tf.zeros(()), trainable=True, name="gamma")

    def call_global(self, states, global_state):
        """Equation 12.

        Global RNN. Inputs are prepared (except for final mean) in ``call``.
        """
        # [1, units] -> [num tensors, 1, units] -> [1, units]
        inputs = tf.reduce_mean(tf.stack(
            [state["tensor"] for state in states]), 0)
        global_state_new, _ = self.global_rnn(inputs, global_state)
        return global_state_new

    def _new_momentum_variance(self, grads, states, states_new):
        """Equation 1, 2, 3, 13.

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
        states_new["scaling"] = [
            rms_momentum(
                grads, g_bar, lambda_,
                beta_1=beta_g**(0.5**s), beta_2=beta_lambda**(0.5**s))
            for s, (g_bar, lambda_) in enumerate(states["scaling"])
        ]

        # Scaled momentum
        _m = [
            g_bar / tf.sqrt(lambda_ + self.epsilon)
            for g_bar, lambda_ in states_new["scaling"]
        ]

        # m_t: [timescales, *var shape] -> [var size, timescales]
        return tf.transpose(tf.reshape(tf.stack(_m), [self.timescales, -1]))

    def _relative_log_gradient_magnitude(self, states, states_new):
        """Equation 4.

        Helper function for relative log gradient magnitudes
        """
        log_lambdas = tf.math.log(
            tf.stack([lambda_ for g_bar, lambda_ in states_new["scaling"]])
            + self.epsilon)
        _gamma = log_lambdas - tf.reduce_mean(log_lambdas, axis=0)

        # gamma_t: [timescales, *var shape] -> [var size, timescales]
        return tf.transpose(tf.reshape(_gamma, [self.timescales, -1]))

    def _parameterized_change(self, param, states, states_new, m):
        """Equation 5, 7, 8.

        Helper function for parameter change explicitly parameterized into
        direction and learning rate

        Notes
        -----
        (1) Direction is no longer explicitly parameterized, as specified by
            appendix D.3 in Wichrowska et al.
        (2) A shortcut connection is include as per appendix B.1.
        """
        # New learning rate
        # Eq 7, 8
        d_eta = tf.reshape(self.delta_nu(states_new["param"]), tf.shape(param))
        eta = d_eta + states["eta_bar"]
        sg = tf.nn.sigmoid(self.gamma)
        states_new["eta_bar"] = (sg * states["eta_bar"] + (1 - sg) * eta)

        # Relative log learning rate
        # Eq Unnamed, end of sec 3.2.4
        states_new["eta_rel"] = tf.reshape(
            eta - tf.math.reduce_mean(eta), [-1, 1])

        # Direction
        # Eq 5, using the update given in Appendix D.3
        d_theta = tf.reshape(
            self.d_theta(states_new["param"]) + self.gradient_shortcut(m),
            tf.shape(param))
        return tf.exp(eta) * d_theta

    def call(self, param, grads, states, global_state):
        """Optimizer Update.

        Notes
        -----
        The state indices in Wichrowska et al. are incorrect, and should be:
        (1) g_bar^n, lambda^n = EMA(g_bar^n-1, g^n), EMA(lambda^n-1, g^n)
            instead of EMA(..., g^n-1), etc
        (2) h^n = RNN(x^n, h^n-1) instead of h^n+1 = RNN(x^n, h^n)
        Then, the g^n -> g_bar^n, lambda^n -> m^n -> h^n -> d^n data flow
        occurs within the same step instead of across 2 steps. This fix is
        reflected in the original Scale code.

        In order to reduce state size, the state update computation is split:
        (1) Compute beta_g, beta_lambda, m.
        (2) Update Parameter & Tensor RNN.
        (3) Compute eta, d. This step only depends on the parameter RNN,
            so the Global RNN being updated after this does not matter.
        (4) Update Global RNN.
        eta_rel is the only "transient" (i.e. not RNN hidden states, momentum,
        variance, learning rate) product stored in the optimizer state.
        """
        states_new = {}

        # Prerequisites ("Momentum and variance at various timescales")
        # Eq 1, 2, 3, 13
        m = self._new_momentum_variance(grads, states, states_new)
        # Eq 4
        gamma = self._relative_log_gradient_magnitude(states, states_new)

        # Param RNN
        # inputs = [var size, features]
        param_in = tf.concat([
            # x^n:
            m, gamma, states["eta_rel"],
            # h_tensor: [1, hidden size] -> [var size, hidden size]
            tf.tile(states["tensor"], [tf.size(param), 1]),
            # h_global: [1, hidden size] -> [var size, hidden size]
            tf.tile(global_state, [tf.size(param), 1]),
        ], 1)

        # RNN Update
        # Eq 10
        states_new["param"], _ = self.param_rnn(param_in, states["param"])
        # Eq 11
        tensor_in = tf.concat([
            tf.math.reduce_mean(states_new["param"], 0, keepdims=True),
            global_state
        ], 1)
        states_new["tensor"], _ = self.tensor_rnn(tensor_in, states["tensor"])

        # Eq 5, 7, 8
        delta_theta = self._parameterized_change(param, states, states_new, m)

        return delta_theta, states_new

    def get_initial_state(self, var):
        """Get initial model state as a dictionary."""
        batch_size = tf.size(var)

        return {
            "scaling": [
                (tf.zeros(tf.shape(var)), tf.zeros(tf.shape(var)))
                for s in range(self.timescales)],
            "param": self.param_rnn.get_initial_state(
                batch_size=batch_size, dtype=tf.float32),
            "tensor": self.tensor_rnn.get_initial_state(
                batch_size=1, dtype=tf.float32),
            "eta_bar": tf.exp(tf.random.uniform(
                shape=tf.shape(var),
                minval=tf.math.log(self.init_lr[0]),
                maxval=tf.math.log(self.init_lr[1]))),
            "eta_rel": tf.zeros([batch_size, 1]),
        }

    def get_initial_state_global(self):
        """Initialize global hidden state."""
        return self.global_rnn.get_initial_state(
            batch_size=1, dtype=tf.float32)
