"""Curriculum Learning Training Strategy."""
import os
import numpy as np

from .strategy import BaseStrategy
from .deserialize import to_float_schedule, to_integer_schedule


class CurriculumLearningStrategy(BaseStrategy):
    """Curriculum Learning Training Manager.

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
    unroll_schedule : callable(int) -> int or int[] or dict
        callable: callable that obtains the unroll size for training stage i.
        int[]: list of unroll lengths specified explicitly. Will limit
            max_stages to its length.
        dict: unroll length in the form of N_i = ["base"] * ["power"]^i
    epoch_schedule : callable(int) -> int or int[] or dict
        callable: callable that obtains the number of epochs to run for each
            problem for training stage i.
        int[]: list of epoch depths specified explicitly.
        dict: epoch depth in the form of N_i = ["base"] * ["power"]^i
    annealing_schedule : callable(int) -> float or float or float[]
        callable: function returning the probability of choosing imitation
            learning for a given period. The idea is to anneal this to 0.
        float: sets annealing_schedule ~ exp(-i * x)
        float[]: specify the annealing schedule explicitly as a list or tuple.
    """

    COLUMNS = {
        'period': int,
        'stage': int,
        'is_improving': bool,
        'unroll_len': int,
        'validation_len': int,
        'p_teacher': float,
    }

    def __init__(
            self, *args, min_periods=100, max_stages=0,
            unroll_schedule=lambda i: 32 * (2**i),
            epoch_schedule=lambda i: 5 * (2**i),
            annealing_schedule=lambda i: np.exp(i * -0.5),
            name="CurriculumLearningStrategy", **kwargs):

        self.unroll_schedule = to_integer_schedule(
            unroll_schedule, name="unroll")
        self.epoch_schedule = to_integer_schedule(
            epoch_schedule, name="epoch")
        self.annealing_schedule = to_float_schedule(
            annealing_schedule, name="annealing")

        self.min_periods = min_periods
        self.max_stages = max_stages

        super().__init__(*args, name=name, **kwargs)

    def _path(self, stage, period):
        """Get saved model file path."""
        return os.path.join(
            self.directory,
            "stage_{}".format(stage), "period_{}".format(period))

    def _resume(self):
        """Resume current optimization."""
        # Current most recent stage & period
        self.stage = self.summary["stage"].max()
        self.period = self._filter(stage=self.stage)["period"].max()
        self.period += 1

        # Not improving, and past minimum periods
        last_row = self._lookup(stage=self.stage, period=self.period - 1)
        if not last_row["is_improving"] and self.period >= self.min_periods:
            self.stage += 1
            self.period = 0

    def _start(self):
        """Start new optimization."""
        self.stage = 0
        self.period = 0

    def _get_best_loss(self):
        """Get the current validation loss baseline."""
        # First stage
        if self.stage == 0:
            # First period -> best_loss is np.inf & don't load
            if self.period == 0:
                print("First training run; weights initialized from scratch.")
                return np.inf
            # Not the first -> resume from most recent, use previous best loss
            else:
                self._load_network(self.stage, self.period - 1)
                return self._filter(stage=self.stage)["validation_loss"].min()
        # Not first stage
        else:
            # Find best validation loss from previous period
            row_idx = self._filter(
                stage=self.stage - 1)["validation_loss"].idxmin()
            period_idx = self.summary["period"][row_idx]
            # Load & Validate
            self._load_network(self.stage - 1, period_idx)
            print("Validating Best L2O from Stage {}:".format(self.stage - 1))
            best_previous = np.mean(self._run_training_loop(
                self.validation_problems,
                unroll_len=lambda: self.unroll_schedule(self.stage + 1),
                epochs=self.epoch_schedule(self.stage + 1),
                validation=False, p_teacher=0))

            # First period -> use previous best
            if self.period == 0:
                return best_previous
            # Not the first -> resume from most recent
            else:
                self._load_network(self.stage, self.period - 1)
                return min(
                    best_previous,
                    self._filter(stage=self.stage)["validation_loss"].min())

    def learning_stage(self):
        """Learn for a single stage.

        If a stage is partially completed, this method will continue from
        where it last left off based on the contents of summary.csv.
        """
        header = "  {} Stage {}: unroll={}, validation={}  ".format(
            "Starting" if self.period == 0 else "Resuming", self.stage,
            self.unroll_schedule(self.stage),
            self.unroll_schedule(self.stage + 1))
        print("\n" + "-" * len(header))
        print(header)
        print("-" * len(header) + "\n")

        best_loss = self._get_best_loss()

        # Train for at least ``min_periods`` or until we stop improving
        is_improving = True
        while (self.period < self.min_periods) or (is_improving):
            # Learn
            p_teacher = self.annealing_schedule(
                self.stage * self.min_periods + self.period)
            unroll_len = self.unroll_schedule(self.stage)
            validation_len = self.unroll_schedule(self.stage + 1)
            print("\n--- Stage {}, Period {} ---".format(
                self.stage, self.period))
            print("p_teacher={} unroll_len={} validation_len={}".format(
                p_teacher, unroll_len, validation_len))
            results = self._learning_period(
                {"unroll_len": lambda: unroll_len,
                 "epochs": self.epoch_schedule(self.stage),
                 "p_teacher": p_teacher},
                {"unroll_len": lambda: validation_len,
                 "epochs": self.epoch_schedule(self.stage + 1),
                 "p_teacher": 0})

            # Check for improvement
            is_improving = results.validation_loss < best_loss
            if is_improving:
                best_loss = results.validation_loss

            # Save optimizer
            self._save_network(self.stage, self.period)
            # Add to summary
            self._append(
                results, stage=self.stage, period=self.period,
                is_improving=is_improving, p_teacher=p_teacher,
                unroll_len=unroll_len, validation_len=validation_len)
            # Finally increment in memory
            self.period += 1

        # Increment stage
        self.stage += 1
        self.period = 0

    def train(self):
        """Start or resume training."""
        while True:
            self.learning_stage()

            # No longer improving
            is_improving = self._filter(
                stage=self.stage - 1)["is_improving"].any()
            if self.stage > 1 and (not is_improving):
                print("Stopped: no longer improving.")
                break
            # Past specified maximum
            if self.max_stages > 0 and self.stage >= self.max_stages:
                print("Stopped: reached max_stages specification.")
                break
