"""Trainable optimizer base class extending tf.keras.optimizers.Optimizer."""

import tensorflow as tf

from .tf_utils import _var_key
from .utils import wrap_variables, nested_assign


class TrainableOptimizer(tf.keras.optimizers.Optimizer):
    """Trainable optimizer using keras' optimizer API.

    Parameters
    ----------
    network : tf.keras.Model
        Model containing the necessary call methods to operate the optimizer.

    Keyword Args
    ------------
    name : str
        Optimizer name
    weights_file : str | None
        Optional filepath to load optimizer network weights from.
    """

    def __init__(self, network, name="TrainableOptimizer", weights_file=None):

        super().__init__(name)
        self.name = name
        self.network = network
        if weights_file is not None:
            network.load_weights(weights_file)
        self._state_dict = {}

    def add_state(self, var, value):
        """Add state corresponding to a given variable for optimization.

        Parameters
        ----------
        var : tf.Variable
            Variable that the state corresponds to
        value : object
            Nested structure of tensors to initialize state with
        """
        if _var_key(var) not in self._state_dict:
            value = wrap_variables(value, trainable=False)
            self._state_dict[_var_key(var)] = value

    def get_state(self, var):
        """Get current state nested structure variables.

        Parameters
        ----------
        var : tf.Variable
            Variable to look up

        Returns
        -------
        object
            Nested structure of variables corresponding to ``var``.
        """
        return self._state_dict[_var_key(var)]

    def _create_slots(self, var_list):
        """Create slots function required by tf.keras.optimizers.Optimizer.

        Parameters
        ----------
        var_list : tf.Variable[]
            List of variables to be optimized on; passed by parent.
        """
        for var in var_list:
            self.add_state(var, self._initialize_state(var))

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
        """Optimizer does not use the default global state API."""
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
            Ignored.

        Returns
        -------
        tf.Operation
            Tensorflow operation that assigns new values to the variable and
            defines dependencies (used for control flow)
        """
        state = self.get_state(var)
        new_var, new_state = self._compute_update(var, grad, state)

        # Track ops for tf.group
        ops = nested_assign(state, new_state)
        ops.append(var.assign(new_var))
        return tf.group(ops)

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
        """Serialize Configuration; passes through to network."""
        return self.network.get_config()

    def variables(self):
        """Returns variables of this Optimizer based on the order created.

        Override of base method to use _state_dict instead of _weights.
        """
        return tf.nest.flatten(self._state_dict)
