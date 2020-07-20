"""CoordinateWise Optimizer Architecture"""

from . trainable_optimizer import TrainableOptimizer


class CoordinateWiseOptimizer(TrainableOptimizer):
    """Coordinatewise Optimizer as described in
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
        super().__init__(name, **kwargs)

    def _initialize_state(self, var):
        """Fetch initial states from child network."""
        return self.network.get_initial_state(var)

    def _compute_update(self, param, grad, state):
        """Compute updates from child network."""
        dparam, new_state = self.network(param, grad, state)
        return param - dparam, new_state
