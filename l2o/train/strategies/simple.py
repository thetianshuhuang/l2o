import os
import numpy as np

from .strategy import BaseStrategy


class SimpleStrategy(BaseStrategy):
    """


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

    COLUMNS = {
        'period': int,
        'training_loss': float,
        'validation_loss': float
    }

    def __init__(
            self, *args, num_periods=100,
            unroll_distribution=lambda: np.random.geometric(0.05),
            annealing_schedule=lambda i: np.exp(i * -0.5),
            validation_unroll=50, name="SimpleStrategy", **kwargs):

        super().__init__(*args, name=name, **kwargs)
        self.num_periods = num_periods
        self.validation_unroll = validation_unroll

        # Deserialize unroll
        if type(unroll_distribution) == float:
            self.unroll_distribution = (
                lambda: np.random.geometric(unroll_distribution))
        elif type(unroll_distribution) == int:
            self.unroll_distribution = lambda: unroll_distribution
        elif callable(unroll_distribution):
            self.unroll_distribution = unroll_distribution
        else:
            raise TypeError(
                "Unrecognized unroll distribution type; must be float or "
                "callable(() -> int)")

        # Deserialize annealing schedule
        if type(annealing_schedule) == float:
            self.annealing_schedule = (
                lambda i: np.exp(i * -np.abs(annealing_schedule)))
        elif type(annealing_schedule) in (list, tuple):
            self.annealing_schedule = annealing_schedule.__getitem__
        elif callable(annealing_schedule):
            self.annealing_schedule = annealing_schedule
        else:
            raise TypeError(
                "Unrecognized annealing_schedule type; must be float or list "
                "or callable(int -> float)")

    def _path(self, period):
        """Get saved model file path"""
        return os.path.join(self.directory, "period_{}".format(period))

    def _resume(self):
        """Resume current optimization."""
        most_recent = self.summary["period"].max()
        self._load_network(most_recent)
        self.period = most_recent + 1

    def _start(self):
        """Resume current optimization."""
        self.period = 0

    def train(self):
        """The actual training method."""

        print(self)
        while self.period < self.num_periods:

            p_teacher = self.annealing_schedule(self.period)
            print("--- Period {} [p_teacher={}] ---".format(
                self.period, p_teacher))

            training_loss, validation_loss = self._learning_period(
                {"unroll_len": self.unroll_distribution,
                 "p_teacher": p_teacher},
                {"unroll_len": lambda: self.validation_unroll,
                 "p_teacher": 0})

            self._save_network(self.period)
            self._append(
                period=self.period, training_loss=training_loss,
                validation_loss=validation_loss)
            self.period += 1
