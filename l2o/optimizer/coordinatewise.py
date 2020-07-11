import tensorflow as tf
from . trainable_optimizer import TrainableOptimizer


class CoordinateWiseOptimizer(TrainableOptimizer):
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
        # First we have to run a dummy computation to trick the network
        # into generating trainable_variables because keras is a bit jank
        self.network(0., 0., self.network.get_initial_state(0.))
        self.trainable_variables = network.trainable_variables

    def _initialize_state(self, var):
        """Fetch initial states from child network."""
        return self.network.get_initial_state(var)

    def _compute_update(self, param, grad, state):
        """Compute updates from child network."""
        dparam, new_state = self.network(param, grad, state)
        return param - dparam, new_state

    def save(self, filepath, **kwargs):
        """Save inernal model using keras model API"""
        self.network.save_weights(filepath, **kwargs)
