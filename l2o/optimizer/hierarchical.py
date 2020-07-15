from .trainable_optimizer import TrainableOptimizer
from .utils import wrap_variables, nested_assign


class HierarchicalOptimizer(TrainableOptimizer):
    def __init__(
            self, network,
            weights_file=None, name="HierarchicalOptimizer", **kwargs):

        super().__init__(name, **kwargs)

        self.network = network
        if weights_file is not None:
            network.load_weights(weights_file)

        init_global = self.network.get_initial_state_global()
        self._state_dict["__global__"] = wrap_variables(init_global())

        # Alias trainable_variables
        # First we have to run a dummy computation to trick the network into
        # generating trainable_variables
        self.network.call(0., 0., self.network.get_initial_state(0.))
        self.network.call_global([], init_global())
        self.trainable_variables = network.trainable_variables

    def _initialize_state(self, var):
        return self.param_rnn.get_initial_state(var)

    def _compute_update(self, param, grad, state):
        dparam, new_state = self.network.call(param, grad, state)
        return param - dparam, new_state

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Overrides apply_gradients in order to call global update."""

        # Eq 10, 11, 13, and prerequisites
        # Calls _compute_update
        super().apply_gradients(grads_and_vars, *args, **kwargs)
        # Eq 12
        global_state = self._state_dict["__global__"]
        nested_assign(
            global_state,
            self.network.call_global(
                [self.get_state(var) for grad, var in grads_and_vars],
                global_state))

    def save(self, filepath, **kwargs):

        self.network.save_weights(filepath, **kwargs)
