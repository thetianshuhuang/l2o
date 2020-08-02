"""Simple Training Strategy."""
import os
import numpy as np

from .strategy import BaseStrategy
from .deserialize import to_integer_distribution, to_float_schedule


class SimpleStrategy(BaseStrategy):
    """Basic Iterative Training Strategy.

    Parameters
    ----------
    *args : object[]
        Arguments passed to BaseStrategy

    Keyword Args
    ------------
    **kwargs : dict
        Arguments passed to BaseStrategy
    num_periods : int
        Number of periods to train for
    unroll_distribution : callable(() -> int) or int or float
        callable: function returning the the unroll length; is rerolled each
            epoch within each training problem.
        int: fixed unroll length.
        float: sets unroll_distribution ~ Geometric(x)
    annealing_schedule : callable(int -> float) or float or float[]
        callable: function returning the probability of choosing imitation
            learning for a given period. The idea is to anneal this to 0.
        float: sets annealing_schedule ~ exp(-i * x)
        float[]: specify the annealing schedule explicitly as a list or tuple.
    validation_unroll : int
        Unroll length to use for validation.
    """

    COLUMNS = {'period': int}

    def __init__(
            self, *args, num_periods=100,
            unroll_distribution=lambda: np.random.geometric(0.05),
            annealing_schedule=lambda i: np.exp(i * -0.5),
            validation_unroll=50, name="SimpleStrategy", **kwargs):

        super().__init__(*args, name=name, **kwargs)
        self.num_periods = num_periods
        self.validation_unroll = validation_unroll

        self.unroll_distribution = to_integer_distribution(
            unroll_distribution, name="unroll")
        self.annealing_schedule = to_float_schedule(
            annealing_schedule, name="annealing")

    def _path(self, period):
        """Get saved model file path."""
        return os.path.join(self.directory, "period_{}".format(period))

    def _resume(self):
        """Resume current optimization."""
        most_recent = self.summary["period"].max()
        self._load_network(most_recent)
        self.period = most_recent + 1

    def _start(self):
        """Start new optimization."""
        self.period = 0

    def train(self):
        """Start or resume training."""
        while self.period < self.num_periods:

            p_teacher = self.annealing_schedule(self.period)
            print("--- Period {} [p_teacher={}] ---".format(
                self.period, p_teacher))

            results = self._learning_period(
                {"unroll_len": self.unroll_distribution,
                 "p_teacher": p_teacher},
                {"unroll_len": lambda: self.validation_unroll,
                 "p_teacher": 0})

            self._save_network(self.period)
            self._append(results, period=self.period)
            self.period += 1
