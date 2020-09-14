"""Base Strategy Class and Utilities."""
import os
import json
import collections

import tensorflow as tf
import numpy as np
import pandas as pd

from l2o import problems
from l2o.evaluate import evaluate
from .deserialize import deserialize_problems, get_optimizer


TrainingPeriod = collections.namedtuple(
    "TrainingPeriod", [
        "meta_loss_mean", "meta_loss",
        "imitation_loss_mean", "imitation_loss",
        "validation_loss"
    ])


def _makedir(path, assert_empty=False):
    """Helper function to create a directory."""
    if os.path.isdir(path):
        if assert_empty:
            raise Exception(
                path, "Directory {} already exists; please rename or "
                "delete the directory.".format(path))
    else:
        os.mkdir(path)


class BaseStrategy:
    """Base Class for training strategies.

    Parameters
    ----------
    learner : optimizer.TrainableOptimizer
        Optimizer to train

    Keyword Args
    ------------
    train_args : dict
        Arguments to pass to ``train``.
    problems : problems.ProblemSpec[]
        List of problem specifications to train on
    problems : problems.ProblemSpec[] or None
        List of problems to validate with. If None, validates on the training
        problem set.
    epochs_per_period : int
        Number of meta-epochs to train per training 'period'
    validation_seed : int
        Seed for optimizee initialization during validation
    optimizer : tf.keras.optimizers.Optimizer or str or dict
        Optimizer to train learned optimizer with; initialized with
        tf.keras.optimizers.get to support str and dict formats.
    directory : str
        Directory to save weights and other data to

    Attributes
    ----------
    COLUMNS : dict
        Dict containing additional summary keys and data types; can be
        overridden to add keys other than 'training_loss',
        'mean_training_loss', and 'validation_loss'.
    """

    COLUMNS = {}

    def __init__(
            self, learner, name="GenericStrategy", train_args={}, problems=[],
            validation_problems=None,
            epochs_per_period=10, validation_seed=12345,
            optimizer="Adam", directory="weights"):

        self.problems = deserialize_problems(problems)
        self.validation_problems = deserialize_problems(
            validation_problems, default=self.problems)

        self.learner = learner
        self.optimizer = get_optimizer(optimizer)
        self.train_args = train_args

        self.epochs_per_period = epochs_per_period
        self.validation_seed = validation_seed

        self.name = name
        self.directory = directory
        _makedir(self.directory)

        try:
            self.summary = pd.read_csv(
                os.path.join(self.directory, "summary.csv"))
            self._resume()
        except FileNotFoundError:
            columns = dict(
                meta_loss_mean=float, imitation_loss_mean=float,
                validation_loss=float, **self.COLUMNS, **{
                    name.format(i): float
                    for name in ["meta_loss_{}", "imitation_loss_{}"]
                    for i in range(self.epochs_per_period)
                })
            self.summary = pd.DataFrame({
                k: pd.Series([], dtype=v) for k, v in columns.items()})
            self._start()

    def __repr__(self):
        """__repr__ override."""
        return "<{} training {}:{} @ {}>".format(
            self.name, self.learner.name,
            self.learner.network.name, self.directory)

    def _path(self, *args, **kwargs):
        """Get saved model file path."""
        raise NotImplementedError()

    def _resume(self):
        """Resume current optimization."""
        raise NotImplementedError()

    def _start(self):
        """Start new optimization."""
        raise NotImplementedError()

    def _append(self, results, **kwargs):
        """Append to summary statistics."""
        period_losses = {
            name.format(i): val
            for name, losses in [
                ("meta_loss_{}", results.meta_loss),
                ("imitation_loss_{}", results.imitation_loss)]
            for i, val in enumerate(losses)
        }
        new_row = dict(
            meta_loss_mean=results.meta_loss_mean,
            imitation_loss_mean=results.imitation_loss_mean,
            validation_loss=results.validation_loss,
            **period_losses, **{k: v for k, v in kwargs.items()})

        self.summary = self.summary.append(new_row, ignore_index=True)
        self.summary.to_csv(
            os.path.join(self.directory, "summary.csv"), index=False)

    def _filter(self, **kwargs):
        """Helper function to filter dataframe."""
        try:
            filtered = self.summary
            for k, v in kwargs.items():
                filtered = filtered[filtered[k] == v]
            return filtered
        except IndexError:
            return None

    def _lookup(self, **kwargs):
        """Helper function to look up values from dataframe."""
        filtered = self._filter(**kwargs)
        if filtered is not None:
            return filtered.iloc[0]
        else:
            return None

    def _load_network(self, *args, **kwargs):
        """Helper function to load network weights and optimizer state."""
        path = self._path(*args, **kwargs)
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.learner.network)
        checkpoint.read(path).expect_partial()
        print("Loaded weights: {}".format(path))

    def _save_network(self, *args, **kwargs):
        """Helper function to save network weights and optimizer state."""
        path = self._path(*args, **kwargs)
        _makedir(os.path.dirname(path))
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.learner.network)
        checkpoint.write(file_prefix=path)
        print("Saved weights: {}".format(path))

    def _run_training_loop(self, problems, **kwargs):
        """Run Training Loop.

        Returns
        -------
        float[2]
            [0] Imitation learning mean loss
            [1] Meta learning mean loss
        """
        args_merged = {**self.train_args, **kwargs}
        results = self.learner.train(problems, self.optimizer, **args_merged)
        imitation_loss, meta_loss = list(zip(*results))
        return np.mean(imitation_loss), np.mean(meta_loss)

    def _learning_period(self, train_args, validation_args):
        """Trains for ``epochs_per_period`` meta-epochs.

        Parameters
        ----------
        train_args : dict
            Arguments to pass to self.learner.train
        validation_args : dict
            Arguments to pass to self.learner.train when validate=True

        Returns
        -------
        TrainingPeriod
            Named tuple, with keywords:
            meta_loss_mean : float
                Mean training loss for meta loss
            imitation_loss_mean : float
                Mean training loss for imitation loss
            meta_loss : float[]
                Training loss for meta loss
            imitation_loss : float[]
                Training loss for imitation loss
            validation_loss : float
                Validation loss
        """
        print("Training:")

        # Train for ``epochs_per_period`` meta-epochs
        training_loss = []
        for i in range(self.epochs_per_period):
            print("Meta-Epoch {}/{}".format(i + 1, self.epochs_per_period))
            training_loss.append(self._run_training_loop(
                self.problems, validation=False, **train_args))
        imitation_loss, meta_loss = list(zip(*training_loss))

        # Compute validation loss
        print("Validating:")
        _, validation_loss = self._run_training_loop(
            self.validation_problems, validation=True, **validation_args)

        results = TrainingPeriod(
            meta_loss_mean=np.mean(meta_loss),
            meta_loss=meta_loss,
            imitation_loss_mean=np.mean(imitation_loss),
            imitation_loss=imitation_loss,
            validation_loss=validation_loss
        )
        print(
            "imitation_loss: {} | meta_loss: {} | validation_loss: {}".format(
                results.imitation_loss_mean, results.meta_loss_mean,
                validation_loss))
        return results

    def train(self):
        """Actual training method."""
        raise NotImplementedError()

    def evaluate(self, *args, repeat=1, save=True, **kwargs):
        """Evaluate L2O.

        Parameters
        ----------
        *args: list
            Passed on to _path(). Should be (period,) for SimpleStrategy and
            (stage, period) for CurriculumLearningStrategy.
        **kwargs : dict
            Passed on to l2o.evaluate.evaluate().

        Keyword Args
        ------------
        repeat : int
            Number of times to repeat the evaluation.
        save : bool
            Save as .npz with the same base name as the weights file?

        Returns
        -------
        dict
            Training results. Can be ignored if save=True. Each key has a shape
            of (repeat, epochs) or (repeat, steps), so an individual trajectory
            should be read as ```results[key][index]```.
        """
        self._load_network(*args)
        results = []
        for i in range(repeat):
            print("Evaluation Training {}/{}".format(i + 1, repeat))
            results.append(
                evaluate(self.learner, **kwargs) for _ in range(repeat))
        results = {k: np.stack([d[k] for d in results]) for k in results[0]}

        if save:
            np.savez(self._path(*args) + '.npz', **results)

        return results
