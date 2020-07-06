import tensorflow as tf

from .loss_mixins import LossMixin
from .tf_utils import _var_key


class TrainableOptimizer(LossMixin, tf.keras.optimizers.Optimizer):

    def __init__(
            self, name,
            use_log_objective=True, obj_train_max_multiplier=-1,
            # use_second_derivatives=True,
            use_numerator_epsilon=False, epsilon=1e-6, **kwargs):
        """Initializes the optimizer with the given name and settings.

        Parameters
        ----------
        name : str
            Optimizer name

        Keyword Args
        ------------
        use_log_objective : bool
            Whether this optimizer uses the logarithm of the objective when
            computing the loss
        obj_train_max_multiplier : float
            The maximum multiplier for the increase in the objective before
            meta-training is stopped. If <= 0, meta-training is not stopped
            early.
        use_second_derivatives : bool (NOT IMPLEMENTED)
            Whether this optimizer uses second derivatives in meta-training.
            This should be set to False if some second derivatives in the
            meta-training problem set are not defined in tensorflow.
        use_numerator_epsilon : bool
            Whether to use epsilon in the numerator when scaling the
            problem objective during meta-training.
        epsilon : float
            Epsilon value.
        """

        # self.use_second_derivatives = use_second_derivatives
        self.use_log_objective = use_log_objective
        self.obj_train_max_multiplier = obj_train_max_multiplier
        self.use_numerator_epsilon = use_numerator_epsilon
        self.epsilon = epsilon

        super(TrainableOptimizer, self).__init__(name)

        self._state_dict = {}

    def assign_state(self, var, value):

        self._state_dict[_var_key(var)] = value

    def get_state(self, var):
        return self._state_dict[_var_key(var)]

    def _create_slots(self, var_list):
        """Create slots function required by tf.keras.optimizers.Optimizer

        Parameters
        ----------
        var_list : tf.Variable[]
            List of variables to be optimized on; passed by parent.
        """
        for var in var_list:
            init_states = self._initialize_state(var)
            # for name, initial in init_states.items():
            # add_slot does nothing if the given var/name are already
            # present. This forces reinitialization.
            # self.add_slot(var, name).assign(initial)
            self.assign_state(var, init_states)

    def _initialize_state(self, var):
        """Initialize any states required for this variable.

        Parameters
        ----------
        var : tf.Variable
            Tensor containing parmeters to be optimized.

        Returns
        -------
        dict
            Keys: str name of each slot;
            Values: tf.Tensor with initial value
        """
        return {}

    def _initialize_global_state(self):
        """todo"""
        return []

    def _resource_apply_dense(self, grad, var, apply_state):
        """Apply optimizer updates to variables.

        Note: this should only get called via _apply_dense or _apply_sparse
        when using the optimizer via optimizer.minimize or
        optimizer.apply_gradients. During meta-training, the optimizer.train
        function should be used to construct an optimization path that is
        differentiable.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor
        var : tf.Variable
            Variable containing weights that grad is computed for. Should have
            the same shape as grad.
        apply_state : dict
            Ignored for now.

        Returns
        -------
        tf.Operation
            Tensorflow operation that assigns new values to the variable and
            defines dependencies (used for control flow)
        """
        # state = {
        # key: self.get_slot(var, key) for key in self.get_slot_names()}
        new_var, new_state = self._compute_update(
            var, grad, self.get_state(var))

        self.assign_state(var, new_state)
        return var.assign_sub(new_var)

    def _resource_update_sparse(self, grad, var):
        raise NotImplementedError()

    def _compute_update(self, param, grad, state):
        """Computes the update step for optimization.

        Parameters
        ----------
        param : tf.Variable
            Variable containing parameters to optimize
        grad : tf.Tensor
            Gradient tensor
        state : dict
            Any extra states required by the optimizer. Keys are strings;
            values are tf.Variable() tracked by add_slot/get_slot

        Returns
        -------
        (tf.Variable, dict)
            [0] : updated parameters
            [1] : updated state variables (same format as `state`)
        """
        raise NotImplementedError()

    def get_config(self):
        return {
            "use_log_objective": self.use_log_objective,
            "obj_train_max_multiplier": self.obj_train_max_multiplier,
            "use_numerator_epsilon": self.use_numerator_epsilon,
            "epsilon": self.epsilon
        }

    def save(self, filepath, **kwargs):
        """Save model to file.

        Parameters
        ----------
        filepath : str
            Destination file
        """
        raise NotImplementedError()
