"""Optimizer Architectures."""

import itertools

from .trainable_optimizer import TrainableOptimizer
from .utils import wrap_variables, nested_assign


class CoordinateWiseOptimizer(TrainableOptimizer):
    """Coordinatewise Optimizer.

    Described in
    "Learing to learn by gradient descent by gradient descent"
    (Andrychowicz et. al, 2016)

    Parameters
    ----------
    network : tf.keras.Model
        Module to apply to each coordinate.

    Keyword Args
    ------------
    name : str
        Optimizer name
    **kwargs : dict
        Passed on to TrainableOptimizer.
    """

    def __init__(self, network, name="CoordinateWiseOptimizer", **kwargs):
        super().__init__(network, name=name, **kwargs)

    def _initialize_state(self, var):
        """Fetch initial states from child network."""
        return self.network.get_initial_state(var)

    def _compute_update(self, param, grad, state):
        """Compute updates from child network."""
        dparam, new_state = self.network(param, grad, state)
        return param - dparam, new_state


class HierarchicalOptimizer(TrainableOptimizer):
    """Hierarchical Optimizer.

    Described in
    "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)

    Parameters
    ----------
    network : tf.keras.Model
        Module to apply to each coordinate.

    Keyword Args
    ------------
    name : str
        Optimizer name
    **kwargs : dict
        Arguments passed to TrainableOptimizer
    """

    def __init__(self, network, name="HierarchicalOptimizer", **kwargs):

        super().__init__(network, name=name, **kwargs)

        # Global state put into the state dict to make .variables() easier
        # This way all state information is contained in _state_dict
        init_global = self.network.get_initial_state_global()
        self._state_dict["__global__"] = wrap_variables(init_global)

    def reset(self):
        """Reset optimizer state.

        Override needed to reset global state while still keeping the same
        variables.
        """
        global_vars = self._state_dict["__global__"]
        nested_assign(global_vars, self.network.get_initial_state_global())
        self._state_dict = {"__global__": global_vars}

    def _initialize_state(self, var):
        """Fetch initial states from child network."""
        return self.network.get_initial_state(var)

    def _compute_update(self, param, grad, state):
        """Fetch initial states from child network."""
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
