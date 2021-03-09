"""Simple Iterative Training Strategy."""

import os
import numpy as np

from .strategy import BaseStrategy
from l2o import deserialize


class SimpleStrategy(BaseStrategy):
    """Simple Iterative Training Strategy.

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
    validation_seed : int
        Seed for optimizee initialization during validation
    directory : str
        Directory to save weights and other data to
    name : str
        Strategy name.
    num_periods : int
        Number of periods to train for
    unroll_len : integer_schedule
        Specifies unroll length for each period.
    depth : integer_schedule
        Specifies number of outer steps per outer epoch for each period
        (number of outer steps before resetting training problem)
    epochs : integer_schedule
        Specifies number of outer epochs to run for each period.
    annealing_schedule : float_schedule
        Specifies p_teacher for each period.
    validation_epochs : int
        Number of outer epochs during validation.
    validation_depth : int
        Depth during validation.
    validation_unroll : int
        Unroll length to use for validation.
    warmup : integer_schedule
        Number of iterations for warmup; if 0, no warmup is applied.
    warmup_rate : float_schedule
        SGD Learning rate during warmup period.
    validation_warmup : int
        Number of iterations for warmup during validation.
    validation_warmup_rate : float
        SGD learning rate during warmup for validation.
    """

    metadata_columns = {
        "period": int,
    }
    hyperparameter_columns = {
        "warmup": int,
        "warmup_rate": float,
        "p_teacher": float,
        "depth": int,
        "unroll_len": int,
        "epochs": int,
    }

    def __init__(
            self, *args, num_periods=100, unroll_len=200, epochs=1,
            depth=1, annealing_schedule=0.1, validation_epochs=None,
            validation_depth=None, validation_unroll=None, warmup=0,
            warmup_rate=0.01, validation_warmup=0, validation_warmup_rate=0.01,
            name="SimpleStrategy", **kwargs):

        self.num_periods = num_periods

        def _default(val, default):
            return val if val is not None else default

        self.validation_epochs = _default(validation_epochs, epochs)
        self.validation_depth = _default(validation_depth, depth)
        self.validation_unroll = _default(validation_unroll, unroll_len)

        self.epochs = deserialize.integer_schedule(epochs, name="epochs")
        self.depth = deserialize.integer_schedule(depth, name="depth")
        self.unroll_len = deserialize.integer_schedule(
            unroll_len, name="unroll")
        self.annealing_schedule = deserialize.float_schedule(
            annealing_schedule, name="annealing")

        self.warmup_schedule = deserialize.integer_schedule(
            warmup, name="warmup")
        self.warmup_rate_schedule = deserialize.float_schedule(
            warmup_rate, name="warmup_rate")
        self.validation_warmup = validation_warmup
        self.validation_warmup_rate = validation_warmup_rate

        super().__init__(*args, name=name, **kwargs)

    def _path(self, period=0, dtype="checkpoint", file="test"):
        """Get file path for the given metadata."""
        return self._base_path("period_{:n}".format(period), dtype)

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

            train_args = {
                "unroll_len": self.unroll_len(self.period),
                "p_teacher": self.annealing_schedule(self.period),
                "warmup": self.warmup_schedule(self.period),
                "warmup_rate": self.warmup_rate_schedule(self.period),
                "depth": self.depth(self.period),
                "epochs": self.epochs(self.period)
            }
            validation_args = {
                "unroll_len": self.validation_unroll,
                "p_teacher": 0,
                "warmup": self.validation_warmup,
                "warmup_rate": self.validation_warmup_rate,
                "depth": self.validation_depth,
                "epochs": self.validation_epochs
            }
            metadata = {"period": self.period}

            hypers = ", ".join([
                "{}={}".format(k, train_args[k])
                for k in self.hyperparameter_columns])
            print("\nPeriod {}: {}".format(self.period, hypers))

            self._training_period(train_args, validation_args, metadata)
            self.period += 1
