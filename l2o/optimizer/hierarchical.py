
from .trainable_optimizer import TrainableOptimizer


class HierarchicalOptimizer(TrainableOptimizer):
    def __init__(
            self, parameter_net, tensor_net, global_net,
            name="HierarchicalOptimizer", **kwargs):
        super().__init__(name, **kwargs)

        self.nets = {
            "param": parameter_net,
            "tensor": tensor_net,
            "global": global_net
        }

    def _initialize_state(self, var):
        return self.nets["param"].get_initial_state(var)

    def _compute_update(self, param, grad, state):
        # todo
        pass

    def apply_gradients(
            self, grads_and_vars, name=None,
            experimental_aggregate_gradients=True):

        # todo
        # can process global updates either before or after if desired

        super().apply_gradients(
            grads_and_vars, name=name,
            experimental_aggregate_gradients=experimental_aggregate_gradients)

    def save(self, filepath, **kwargs):

        for name, net in self.nets.items():
            net.save_weights(filepath + '_' + name, **kwargs)
