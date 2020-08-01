import os
import numpy as np

from .strategy import BaseStrategy


class CurriculumLearningStrategy(BaseStrategy):
    """Curriculum Learning Manager

    Parameters
    ----------
    *args : object[]
        Arguments passed to BaseStrategy

    Keyword Args
    ------------
    **kwargs : dict
        Arguments passed to BaseStrategy
    min_periods : int
        Minimum number of training periods per stage
    max_stages : int
        Maximum number of stages to run. If 0, runs until the validation loss
        stops improving.
    schedule : callable(int -> int) or int[] or dict
        callable: callable that obtains the unroll size for training stage i.
        int[]: list of unroll lengths specified explicitly. Will limit
            max_stages to its length.
        dict: unroll length in the form of N_i = ["base"] * ["power"]^i
    """

    COLUMNS = {
        'period': int,
        'stage': int,
        'is_improving': bool,
    }

    def __init__(
            self, *args, min_periods=100, max_stages=0,
            schedule=lambda i: 50 * (2**i), name="CurriculumLearningStrategy",
            **kwargs):

        super().__init__(*args, name=name, **kwargs)

        # List -> convert to function
        if type(schedule) == list or type(schedule) == tuple:
            self.schedule = schedule.__getitem__
            max_stages = len(self.schedule)
        # Dict -> convert to exponential
        elif type(schedule) == dict:
            self.schedule = lambda i: schedule["base"] * (schedule["power"]**i)
        # Callable
        elif callable(schedule):
            self.schedule = schedule
        else:
            raise TypeError(
                "Unrecognized schedule dtype. Must be list, dict "
                "with keys 'base' (int), 'power' (int), or callable "
                "(int -> int).")

        self.min_periods = min_periods
        self.max_stages = max_stages

    def _path(self, stage, period):
        """Get saved model file path"""
        return os.path.join(
            self.directory,
            "stage_{}".format(stage), "period_{}".format(period))

    def _resume(self):
        """Resume current optimization."""

        # Current most recent stage & period
        self.stage = self.summary["stage"].max()
        self.period = self.summary["period"].max()

        # Not improving, and past minimum periods
        last_row = self._lookup(stage=self.stage, period=self.period)
        if (not last_row["is_improving"]) and self.period >= self.min_periods:
            self.stage += 1
            self.period = 0
        # Same period
        else:
            self.period += 1

    def _start(self):
        """Start new optimization."""
        self.stage = 0
        self.period = 0

    def _get_best_loss(self):
        """Helper function to get the current validation loss baseline."""

        # First period and past first s -> load best from previous
        if self.period == 0 and self.stage > 0:
            # Find best validation loss
            row_idx = self._lookup(
                stage=self.stage - 1)["validation_loss"].idxmin()
            period_idx = self.summary["period"][row_idx]
            # Load & Validate
            self._load_network(self.stage - 1, period_idx)
            return self.learner.train(
                self.problems, self.optimizer, validation=False,
                unroll=self.schedule(self.stage + 1), **self.train_args
            ).validation_loss

        # First period and first stage -> best_loss is np.inf & don't load
        elif self.period == 0 and self.stage == 0:
            return np.inf

        # Not the first -> resume from most recent
        else:
            self._load_network(self.stage, self.period - 1)
            return self._lookup(stage=self.stage)["validation_loss"].min()

    def learning_stage(self):
        """Learn for a single stage.

        If a stage is partially completed, this method will continue from
        where it last left off based on the contents of summary.csv.
        """

        header = "  {} Stage {}: unroll={}, validation={}  ".format(
            "Starting" if self.period == 0 else "Resuming", self.stage,
            self.schedule(self.stage), self.schedule(self.stage + 1))
        print("\n" + "-" * len(header))
        print(header)
        print("-" * len(header) + "\n")

        best_loss = self._get_best_loss()

        # Train for at least ``min_periods`` or until we stop improving
        is_improving = True
        while (self.period < self.min_periods) or (is_improving):
            # Learn
            print("--- Stage {}, Period {} ---".format(
                self.stage, self.period))
            results = self._learning_period(
                {"unroll_len": lambda: self.schedule(self.stage)},
                {"unroll_len": lambda: self.schedule(self.stage + 1)})

            # Check for improvement
            is_improving = results.validation_loss < best_loss
            if is_improving:
                best_loss = results.validation_loss

            # Save optimizer
            self._save_network()
            # Add to summary
            self._append(
                results, stage=self.stage, period=self.period,
                is_improving=is_improving)
            # Finally increment in memory
            self.period += 1

        # Increment stage
        self.stage += 1
        self.period = 0

    def train(self):
        """Start or resume training."""

        print(self)
        while True:
            self.learning_stage()

            # No longer improving
            is_improving = self._lookup(stage=self.stage)["improving"].any()
            if self.stage > 1 and (not is_improving):
                print("Stopped: no longer improving.")
                break
            # Past specified maximum
            if self.max_stages > 0 and self.stage >= self.max_stages:
                print("Stopped: reached max_stages specification.")
                break
