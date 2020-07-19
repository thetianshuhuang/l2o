"""Hierarchical Optimizer Architecture"""

import itertools

from .trainable_optimizer import TrainableOptimizer
from .utils import wrap_variables, nested_assign


class HierarchicalOptimizer(TrainableOptimizer):
    """Hierarchical Optimizer as described in
    "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)

    Parameters
    ----------
    network : tf.keras.Model
        Model containing update methods for coordinate, tensor, and global
        RNNs.
    weights_file : str | None
        Optional filepath to load optimizer network weights from.
    **kwargs : dict
        Passed on to TrainableOptimizer.
    """

    def __init__(
            self, network,
            weights_file=None, name="HierarchicalOptimizer", **kwargs):

        super().__init__(name, **kwargs)

        self.network = network
        if weights_file is not None:
            network.load_weights(weights_file)

        # Global state put into the state dict to make .variables() easier
        # This way all state information is contained in _state_dict
        init_global = self.network.get_initial_state_global()
        self._state_dict["__global__"] = wrap_variables(init_global)

        # Alias trainable_variables
        # First we have to run a dummy computation to trick the network into
        # generating trainable_variables
        local_state = self.network.get_initial_state(0.)
        self.network.call(0., 0., local_state, init_global)
        self.network.call_global([local_state], init_global)
        self.trainable_variables = network.trainable_variables

    def _initialize_state(self, var):
        return self.network.get_initial_state(var)

    def _compute_update(self, param, grad, state):
        dparam, new_state = self.network.call(
            param, grad, state, self._state_dict["__global__"])
        return param - dparam, new_state

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Overrides apply_gradients in order to call global update."""

        # Make copy since grads_and_vars is a zip iterable which is
        # not reusable once super().apply_gradients() sucks it up
        grads_and_vars, grads_and_vars_cpy = itertools.tee(grads_and_vars)

        # Eq 10, 11, 13, and prerequisites
        # Calls _compute_update
        res = super().apply_gradients(grads_and_vars, *args, **kwargs)
        # Eq 12
        nested_assign(
            self._state_dict["__global__"],
            self.network.call_global(
                [self.get_state(var) for grad, var in grads_and_vars_cpy],
                self._state_dict["__global__"]))

        return res

    def save(self, filepath, **kwargs):

        self.network.save_weights(filepath, **kwargs)
