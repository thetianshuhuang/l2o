"""Learned optimizer training strategy."""

import os
import pandas as pd
import numpy as np
import tensorflow as tf

from l2o.train.loss_tracker import LossTracker
from l2o.evaluate import evaluate
from l2o import deserialize


class BaseStrategy:
    """Base class for learned optimizer training strategies.

    Parameters
    ----------
    learner : train.OptimizerTraining
        Optimizer training wrapper.
    problems : problems.Problem[]
        List of problem specifications to train on.

    Keyword Args
    ------------
    validation_problems : problems.Problem[] or None.
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

    Attributes
    ----------
    columns : dict
        Dict containing additional summary keys and data types; can be
        overridden to add keys other than 'training_loss',
        'mean_training_loss', and 'validation_loss'.
    """

    metadata_columns = {}

    def __init__(
            self, learner, problems, validation_problems=None,
            epochs_per_period=10, validation_seed=12345, directory="weights",
            name="BaseStrategy"):

        self.learner = learner

        self.problems = deserialize.problems(problems)
        self.validation_problems = deserialize.problems(
            validation_problems, default=self.problems)

        self.epochs_per_period = epochs_per_period
        self.validation_seed = validation_seed
        self.directory = directory

        try:
            self.summary = pd.read_csv(
                os.path.join(self.directory, "summary.csv"))
            self._resume()
        except FileNotFoundError:
            columns = dict(
                validation=float, **self.metadata_columns,
                **{k: float for k in self.learner.mean_stats})
            self.summary = pd.DataFrame({
                k: pd.Series([], dtype=v) for k, v in columns.items()})
            self._start()

    def __str__(self):
        """As string -> <Strategy:Optimizer:Network @ directory>."""
        return "<{}:{}:{} @ {}>".format(
            self.name, self.learner.name, self.learner.network.name,
            self.directory)

    def _resume(self):
        """Resume current optimization."""
        raise NotImplementedError()

    def _start(self):
        """Start new optimization."""
        raise NotImplementedError()

    def _path(self, **kwargs):
        """Get saved model file path."""
        raise NotImplementedError()

    def _save_network(self, **kwargs):
        """Wrapper for ``self.learner.save_state`` with ``self._path``."""
        self.learner.save_state(self._path(**kwargs))

    def _load_network(self, **kwargs):
        """Wrapper for ``self.learner.load_state`` with ``self._path``."""
        self.learner.load_state(self._path(**kwargs))

    def _append(self, training_stats, validation_stats, metadata):
        """Save training and validation statistics.

        Parameters
        ----------
        training_stats : dict
            Training statistics to append. Values in ``scalar_statistics`` are
            appended to ``summary.csv``; other values are saved in a .npz
            if present.
        validation_stats : dict
            Validation statistics; ``meta`` is saved.
        metadata : dict
            Strategy metadata; also determines saved filepath (if applicable).
        """
        # Save scalar summary values
        new_row = dict(
            validation=validation_stats["meta_loss"], **metadata,
            **{k: training_stats[k] for k in self.learner.mean_stats})
        self.summary = self.summary.append(new_row, ignore_index=True)
        self.summary.to_csv(
            os.path.join(self.directory, "summary.csv"), index=False)

        # Save other values
        if len(self.learner.stack_stats) > 0:
            save_np = {
                k: training_stats[k] for k in self.learner.stack_stats
            }
            np.savez(
                os.path.join(self._path(**metadata), "log.npz"), **save_np)

    def _training_period(
            self, train_args, validation_args, metadata, eval_file=None,
            eval_args={}):
        """Run a single training period.

        1. Train for ``self.epochs_per_period`` outer epochs.
        2. Validate for a single epoch, possibly using different problems and
            settings.
        3. Save network and optimizer state.
        4. Save summary statistics.

        Parameters
        ----------
        train_args : dict
            Arguments for training. ``validation=False`` is forced.
        validation_args : dict
            Arguments for validation. ``validation=True`` and
            ``validation_seed`` are forced.
        metadata : dict
            Strategy metadata for this training period.

        Keyword Args
        ------------
        eval_file : str or None
            If not None, evaluate at the end of every period, saving results
            to ``path(metadata)/eval_file.npz``.
        eval_args : dict
            If eval_file is not None, pass these arguments to evaluate.
        """
        # Train for ``epochs_per_period`` meta-epochs
        print("Training:")
        training_stats = LossTracker()
        for i in range(self.epochs_per_period):
            print("Meta-Epoch {}/{}".format(i + 1, self.epochs_per_period))
            training_stats.append(self.learner.train(
                self.problems, validation=False, **train_args))
        training_stats = training_stats.summarize(
            self.learner.stack_stats, self.learner.mean_stats)

        # Compute validation loss
        print("Validating:")
        validation_stats = self.learner.train(
            self.validation_problems, validation=True,
            seed=self.validation_seed, **validation_args)

        # Save, append data, and print info
        print("imitation: {} | meta: {} | validation: {}".format(
            training_stats["imitation_loss"], training_stats["meta_loss"],
            validation_stats["meta_loss"]))
        self._save_network(**metadata)
        self._append(training_stats, validation_stats, metadata)

        # Evaluate (if applicable)
        if eval_file is not None:
            self.evaluate(metadata=metadata, file=eval_file, **eval_args)

    def evaluate(self, metadata=None, repeat=1, file="eval", **kwargs):
        """Evaluate network.

        Keyword Args
        ------------
        metadata : dict or None
            If ``dict``, load weights specified by the given metadata.
            Otherwise, uses current weights.
        repeat : int
            Number of repetitions.
        file : str or None
            File to save to. If None, does not save (and only returns).
        kwargs : dict
            Additional arguments to pass to ``evaluate.evaluate``.
        """
        self._load_network(**metadata)
        opt = self.learner.network.architecture(
            self.learner.network, name="OptimizerEvaluation")

        results = []
        for i in range(repeat):
            print("Evaluation Training {}/{}".format(i + 1, repeat))
            results.append(evaluate(opt, **kwargs))
        results = {k: np.stack([d[k] for d in results]) for k in results[0]}

        if file is not None:
            np.savez(os.path.join(self._path(**metadata), name), **results)

        return results
