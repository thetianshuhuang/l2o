import tensorflow as tf

from . import trainable_optimizer as opt


class CoordinateWiseOptimizer(opt.TrainableOptimizer):
    """Coordinatewise Optimizer as described by DM

    Parameters
    ----------
    network : tf.keras.Model
        Module to apply to each coordinate.

    Keyword Args
    ------------
    name : str
        Optimizer name
    weights_file : str | None
        Optional filepath to load optimizer network weights from.
    **kwargs : dict
        Passed on to TrainableOptimizer.
    """

    def __init__(
            self, network,
            weights_file=None, name="CoordinateWiseOptimizer", **kwargs):

        super().__init__(name, **kwargs)

        self.network = network
        if weights_file is not None:
            network.load_weights(weights_file)

        # Alias trainable_variables
        self.trainable_variables = network.trainable_variables

    def _initialize_state(self, var):
        """Fetch initial states from child network."""
        return self.network.get_initial_state(var)

    def _compute_update(self, param, grad, state):
        """Compute updates from child network."""
        return self.network(grad, state)

    def save(self, filepath, **kwargs):
        """Save inernal model using keras model API"""
        self.network.save_weights(filepath, **kwargs)


class HierarchicalOptimizer(opt.TrainableOptimizer):
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

        pass

    def save(self, filepath, **kwargs):

        for name, net in self.nets.items():
            net.save_weights(filepath + '_' + name, **kwargs)
