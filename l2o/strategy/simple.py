"""Simple Training Strategy."""

import os

from .strategy import BaseStrategy
from l2o import deserialize


class SimpleStrategy(BaseStrategy):
    """Basic Iterative Training Strategy.

    Parameters
    ----------
    learner : train.OptimizerTraining
        Optimizer training wrapper.
    problems : problems.ProblemSpec[]
        List of problem specifications to train on.

    Keyword Args
    ------------
    validation_problems : problems.ProblemSpec[] or None.
        List of problems to validate with. If None, validates on the training
        problem set.
    epochs_per_period : int
        Number of meta-epochs to train per training period
    validation_seed : int
        Seed for optimizee initialization during validation
    directory : str
        Directory to save weights and other data to
    name : str
        Strategy name.
    num_periods : int
        Number of periods to train for
    unroll_distribution : callable(() -> int) or int or float
        callable: function returning the the unroll length; is rerolled each
            epoch within each training problem.
        int: fixed unroll length.
        float: sets unroll_distribution ~ Geometric(x)
    depth : int
        Number of outer steps per outer epoch (number of outer steps
        before resetting training problem)
    epochs : int
        Number of outer epochs to run.
    annealing_schedule : callable(int -> float) or float or float[]
        callable: function returning the probability of choosing imitation
            learning for a given period. The idea is to anneal this to 0.
        float: sets annealing_schedule ~ exp(-i * x)
        float[]: specify the annealing schedule explicitly as a list or tuple.
    validation_epochs : int
        Number of outer epochs during validation.
    validation_depth : int
        Depth during validation.
    validation_unroll : int
        Unroll length to use for validation.
    """

    metadata_columns = {"period": int}

    def __init__(
            self, *args, num_periods=100, unroll_distribution=200, epochs=1,
            depth=1, annealing_schedule=0.1, validation_epochs=None,
            validation_depth=None, validation_unroll=None,
            name="SimpleStrategy", **kwargs):

        super().__init__(*args, name=name, **kwargs)

        self.num_periods = num_periods

        def _default(val, default):
            return val if val is not None else default

        self.validation_epochs = _default(validation_epochs, epochs)
        self.validation_depth = _default(validation_depth, depth)
        validation_unroll = _default(validation_unroll, unroll_distribution)

        self.epochs = epochs
        self.depth = depth

        self.validation_unroll = deserialize.integer_distribution(
            validation_unroll, name="validation_unroll")
        self.unroll_distribution = deserialize.integer_distribution(
            unroll_distribution, name="unroll")
        self.annealing_schedule = deserialize.float_schedule(
            annealing_schedule, name="annealing")

    def _path(self, period=0):
        """Get file path for the given metadata."""
        return os.path.join(self.directory, "period_{}".format(int(period)))

    def _resume(self):
        """Resume current optimization."""
        self.period = int(self.summary["period"].max() + 1)

    def _start(self):
        """Start new optimization."""
        self.period = 0

    def train(self):
        """Start or resume training."""
        if self.period > 0:
            self._load_network(period=self.period - 1)

        while self.period < self.num_periods:

            p_teacher = self.annealing_schedule(self.period)
            print("--- Period {} [p_teacher={}] ---".format(
                self.period, p_teacher))

            train_args = {
                "unroll_len": self.unroll_distribution, "p_teacher": p_teacher,
                "depth": self.depth, "epochs": self.epochs}
            validation_args = {
                "unroll_len": self.validation_unroll, "p_teacher": 0,
                "depth": self.validation_depth,
                "epochs": self.validation_epochs}
            metadata = {"period": self.period}

            self._training_period(train_args, validation_args, metadata)
            self.period += 1
