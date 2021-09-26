"""Curriculum Learning Strategy."""

import os
import numpy as np

from .strategy import BaseStrategy
from l2o import deserialize


class CurriculumLearningStrategy(BaseStrategy):
    """Curriculum Learning Strategy.

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
    num_stages : int
        Number of curriculum stages.
    num_periods : int
        Minimum number of periods per stage.
    num_chances : int
        Number of tries to decrease best validation loss before giving up.
    unroll_len : integer_schedule
        Specifies unroll length for each stage.
    depth : integer_schedule
        Specifies number of outer steps per outer epoch for each stage.
    epochs : integer_schedule
        Specifies number of outer epochs to run for each period.
    annealing_schedule : float_schedule
        Specifies p_teacher for each stage.
    validation_epochs : int
        Number of outer epochs during validation.
    warmup : integer_schedule
        Number of iterations for warmup; if 0, no warmup is applied.
    warmup_rate : float_schedule
        SGD Learning rate during warmup period.
    max_repeat : int
        Maximum number of times to repeat period if loss explodes. If
        max_repeat is 0, will repeat indefinitely.
    repeat_threshold : float
        Threshold for repetition, as a proportion of the absolute value of the
        current meta loss.
    """

    metadata_columns = {
        "stage": int,
        "period": int,
        "repeat": int
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
            self, *args, num_stages=5, num_periods=10, num_chances=3,
            unroll_len=200, epochs=1, depth=1, annealing_schedule=0.1,
            validation_epochs=None, max_repeat=0, repeat_threshold=0.1,
            warmup=0, warmup_rate=0.01, name="CurriculumLearningStrategy",
            **kwargs):

        self.num_stages = num_stages
        self.num_periods = num_periods
        self.num_chances = num_chances

        def _default(val, default):
            return val if val is not None else default

        self.epochs = deserialize.integer_schedule(epochs, name="epochs")
        self.depth = deserialize.integer_schedule(depth, name="depth")
        self.unroll_len = deserialize.integer_schedule(
            unroll_len, name="unroll")
        self.annealing_schedule = deserialize.float_schedule(
            annealing_schedule, name="annealing")

        self.max_repeat = max_repeat
        self.repeat_threshold = repeat_threshold

        self.warmup_schedule = deserialize.integer_schedule(
            warmup, name="warmup")
        self.warmup_rate_schedule = deserialize.float_schedule(
            warmup_rate, name="warmup_rate")

        self.validation_epochs = _default(validation_epochs, epochs)
        self.validation_warmup = self.warmup_schedule(self.num_stages)
        self.validation_warmup_rate = self.warmup_rate_schedule(
            self.num_stages)
        self.validation_unroll = self.unroll_len(self.num_stages)

        super().__init__(*args, name=name, **kwargs)

    def _check_repeat(self):
        """Check if current period should be repeated."""
        # First, check repeat limits:
        p_current = self._get(**self._complete_metadata(
            {"stage": self.stage, "period": self.period}))
        if self.max_repeat > 0 and p_current["repeat"] >= self.max_repeat:
            return False

        # Exception to never repeat first checkpoint
        if self.period == 0 and self.stage == 0:
            return False

        # Loss-based checks
        # First period of stage -> use previous stage best train loss
        if self.period == 0:
            loss_ref = self._get(**self._complete_metadata(
                {"stage": self.stage - 1}))["validation"]
            loss_current = p_current["meta_loss"]
        # Not first period -> use current stage best validation loss
        else:
            loss_ref = self._get(**self._complete_metadata(
                {"stage": self.stage, "period": self.period - 1})
            )["validation"]
            loss_current = p_current["validation"]

        max_loss = np.abs(loss_ref) * self.repeat_threshold + loss_ref
        return max_loss < loss_current

    def _continue_stage(self):
        """Check if current stage should continue."""
        if self.period < self.num_periods:
            return True

        stage_best = self.summary.iloc[
            self._filter(stage=self.stage)["validation"].idxmin()]
        return self.period - stage_best["period"] < self.num_chances

    def _complete_metadata(self, metadata):
        """Complete metadata with strategy-dependent fields."""
        metadata = metadata.copy()
        if "stage" not in metadata:
            metadata["stage"] = self.stage
        if "period" not in metadata:
            metadata["period"] = self.summary.iloc[
                self._filter(stage=metadata["stage"])["validation"].idxmin()
            ]["period"]
        if "repeat" not in metadata:
            metadata["repeat"] = self.summary.iloc[
                self._filter(
                    stage=metadata["stage"], period=metadata["period"]
                )["validation"].idxmin()
            ]["repeat"]
        return metadata

    def _load_previous(self):
        """Load network from previous period for resuming or repeating."""
        if self.period == 0:
            self._load_network(**self._complete_metadata(
                {"stage": self.stage - 1}))
        else:
            self._load_network(**self._complete_metadata(
                {"stage": self.stage, "period": self.period - 1}))

    def _path(
            self, stage=0, period=0, repeat=0,
            dtype="checkpoint", file="test"):
        """Get file path for saved data."""
        return self._base_path(
            "stage_{:n}.{:n}.{:n}".format(stage, period, repeat),
            dtype, file=file)

    def _resume(self):
        """Resume current optimization."""
        self.stage = int(self.summary["stage"].max())
        self.period = int(self._filter(stage=self.stage)["period"].max())
        self.repeat = int(self._filter(
            stage=self.stage, period=self.period)["repeat"].max())

        if self._check_repeat():
            self.repeat += 1
        else:
            self.period += 1
            self.repeat = 0

    def _start(self):
        """Start new optimization."""
        self.stage = 0
        self.period = 0
        self.repeat = 0

    def train(self):
        """Start or resume training."""
        if self.period > 0:
            self._load_previous()

        while self.stage < self.num_stages:

            train_args = {
                "unroll_len": self.unroll_len(self.stage),
                "p_teacher": self.annealing_schedule(self.stage),
                "warmup": self.warmup_schedule(self.stage),
                "warmup_rate": self.warmup_rate_schedule(self.stage),
                "depth": self.depth(self.stage),
                "epochs": self.epochs(self.stage)
            }
            validation_args = {
                "unroll_len": self.unroll_len(self.stage + 1),
                "p_teacher": 0,
                "warmup": self.warmup_schedule(self.stage + 1),
                "warmup_rate": self.warmup_rate_schedule(self.stage + 1),
                "depth": self.depth(self.stage + 1),
                "epochs": self.epochs(self.stage + 1)
            }
            metadata = {
                "stage": self.stage,
                "period": self.period,
                "repeat": self.repeat
            }

            hypers = ", ".join([
                "{}={}".format(k, train_args[k])
                for k in self.hyperparameter_columns])
            print("\nStage {}.{}.{}: {}".format(
                self.stage, self.period, self.repeat, hypers))

            self._training_period(train_args, validation_args, metadata)

            # Stage/Period/Repeat logic
            if self._check_repeat():
                self.repeat += 1
                self._load_previous()
            elif self._continue_stage():
                self.period += 1
                self.repeat = 0
            else:
                self._load_network(
                    **self._complete_metadata({"stage": self.stage}))
                self.stage += 1
                self.period = 0
                self.repeat = 0
