"""
Loss args:
- unroll_weights
- teachers
- imitation_optimizer
- strategy
- p_teacher
- epochs
- repeat
- persistent
"""

import os
import functools

import tensorflow as tf
import numpy as np
import pandas as pd


def makedir(path, assert_empty=False):
    if os.path.isdir(path):
        if assert_empty:
            raise Exception(
                path, "Directory {} already exists; please rename or delete "
                "the directory.".format(path))
    else:
        os.mkdir(path)


class CurriculumLearning:
    """Curriculum Learning Manager

    While network weights and learning progress are saved on disk after every
    training period, the caller is responsible for ensuring that the same
    hyper-parameters are passed to the initializer when resumed.

    Parameters
    ----------
    learner : optimizer.TrainableOptimizer
        Training target

    Keyword Args
    ------------
    loss_args : dict
        Arguments to pass to TrainableOptimizer.train. See
        ``train_mixins.TrainingMixin.train`` for documentation.
    problems : problem.ProblemSpec[]
        List of problems to train on
    optimizer : tf.keras.optimizers.Optimizer
        Training optimizer
    min_periods : int
        Minimum number of training periods per stage
    epochs_per_period : int
        Number of meta-epochs per training period
    max_stages : int
        Maximum number of stages to run. If 0, runs until the validation loss
        stops improving.
    directory : str
        Directory containing saved network weights and summary.
    schedule : callable(int -> int)
        Callable that obtains the unroll size for training stage i.
    """

    def __init__(
            self, learner, loss_args={},
            problems=[], optimizer=tf.keras.optimizers.Adam(),
            min_periods=100, epochs_per_period=10, max_stages=0,
            directory="weights",
            schedule=lambda i: 50 * (2**i)):

        self.learner = learner
        self.loss_args = loss_args

        self.problems = problems
        self.optimizer = optimizer
        self.loss_args = loss_args

        self.min_periods = min_periods
        self.epochs_per_period = epochs_per_period
        self.max_stages = max_stages
        self.directory = directory
        self.schedule = schedule

        # Set up directory & summary file
        makedir(self.directory)
        try:
            self.summary = pd.read_csv(
                os.path.join(self.directory, "summary.csv"))
            self.stage = self.summary["stage"].max()
            self.period = self.summary["period"].max() + 1
            self.best_loss = self.summary[
                self.summary["stage"] == self.stage
            ]["validation_loss"].min()
        except FileNotFoundError:
            self.summary = pd.DataFrame({
                "stage": pd.Series([], dtype='int'),
                "period": pd.Series([], dtype='int'),
                "training_loss": pd.Series([], dtype='float'),
                "validation_loss": pd.Series([], dtype='float')
            })
            self.stage = 0
            self.period = 0
            self.best_loss = np.inf

    def _mean_loss(self, results):
        """Helper function to compute mean loss."""
        return np.mean([
            np.mean([np.mean(loss_array) for loss_array in result.loss])
            for result in results
        ])

    def _best_loss(self, stage):
        """Helper function to get the best loss at a given stage."""
        val = self.summary[self.summary["stage"] == stage]["validation_loss"]
        return val.min()

    def learning_period(self):
        """Learn for a single period.

        Returns
        -------
        [float, float]
            [0] mean training loss
            [1] mean validation loss
        """

        # Problems, optimizer, other arguments do not change (only unroll)
        train_func = functools.partial(
            self.learner.train,
            self.problems, self.optimizer, **self.loss_args)

        # Train for ``epochs_per_period`` meta-epochs
        training_loss = []
        for i in range(self.epochs_per_period):
            print("Training [epoch {}/{}]".format(
                i + 1, self.epochs_per_period))
            results = train_func(
                unroll_len=lambda: self.schedule(self.stage), validation=False)
            training_loss.append(self._mean_loss(results))
        training_loss = np.mean(training_loss)

        print("Validating")

        # Compute validation with longer unroll
        validation_loss = self._mean_loss(train_func(
            unroll_len=lambda: self.schedule(self.stage + 1), validation=True))

        print("training_loss: {} | validation_loss: {}".format(
            training_loss, validation_loss))

        return training_loss, validation_loss

    def learning_stage(self):
        """Learn for a single stage.

        If a stage is partially completed, this method will continue from
        where it last left off based on the contents of summary.csv.
        """

        header = "  {} Stage {}: unroll={}, validation={}  ".format(
            "Starting" if self.period == 0 else "Resuming",
            self.stage, self.schedule(self.stage),
            self.schedule(self.stage + 1))
        print("\n" + "-" * len(header))
        print(header)
        print("-" * len(header) + "\n")

        makedir(os.path.join(self.directory, "stage_{}".format(self.stage)))

        # Train for at least ``min_periods`` or until we stop improving
        is_improving = True
        while (self.period < self.min_periods) or (is_improving):
            is_improving = False

            print("--- Stage {}, Period {} ---".format(
                self.stage, self.period))

            training_loss, validation_loss = self.learning_period()

            # Save optimizer
            self.learner.save(os.path.join(
                self.directory,
                "stage_{}".format(self.stage),
                "period_{}".format(self.period)))
            # Add to summary
            self.summary = self.summary.append(
                pd.DataFrame({
                    "stage": [self.stage], "period": [self.period],
                    "training_loss": [training_loss],
                    "validation_loss": [validation_loss]
                }), ignore_index=True)
            self.summary.to_csv(
                os.path.join(self.directory, "summary.csv"), index=False)
            # Finally increment in memory
            self.period += 1

            # Check for improvement
            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                is_improving = True

        self.stage += 1
        self.period = 0

    def train(self):
        """Start or resume training."""

        print("Curriculum Learning")
        print("Training {}:{} @ {}".format(
            self.learner.name, self.learner.network.name, self.directory))
        print("min_periods={}".format(self.min_periods))
        print("epochs_per_period={}".format(self.epochs_per_period))
        print("max_stages={}".format(self.max_stages))

        while True:
            self.learning_stage()

            # No longer improving
            best_now = self._best_loss(self.stage - 1)
            best_past = self._best_loss(self.stage - 2)
            if self.stage > 1 and (best_now >= best_past):
                break
            # Past specified maximum
            if self.max_stages > 0 and self.stage >= self.max_stages:
                break
