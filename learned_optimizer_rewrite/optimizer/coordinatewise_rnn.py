
from . import trainable_optimizer as opt


class CoordinateWiseOptimizer(opt.TrainableOptimizer):
    def __init__(self, network, name="Coordinatewise Optimizer", **kwargs):

        super().__init__(name, **kwargs)
        self.network = network

    def _initialize_state(self, var):

        return self.network.get_initial_state(var)

    def _compute_update(self, param, grad, state):

        return self.network(grad, state)


class HierarchicalOptimizer(opt.TrainableOptimizer):
    def __init__(
            self, parameter_net, tensor_net, global_net,
            name="HierarchicalOptimizer", **kwargs):
        super().__init__(name, **kwargs)
        self.parameter_net = parameter_net
        self.tensor_net = tensor_net
        self.global_net = global_net

    def _initialize_state(self, var):
        return self.parameter_net.get_initial_state(var)

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

        pass