"""Learned optimizer training strategy."""

import os
import time
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
    hyperparameter_columns = {}

    def __init__(
            self, learner, problems, validation_problems=None,
            validation_seed=12345, directory="weights", name="BaseStrategy"):

        self.learner = learner
        self.name = name

        self.problems = deserialize.problems(problems)
        self.validation_problems = deserialize.problems(
            validation_problems, default=self.problems)

        self.validation_seed = validation_seed
        self.directory = directory

        try:
            self.summary = pd.read_csv(
                os.path.join(self.directory, "summary.csv"))
            self._resume()
        except FileNotFoundError:
            columns = dict(
                validation=float, time=float, duration=float,
                **self.metadata_columns,
                **self.hyperparameter_columns,
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

    def _base_path(self, base, dtype, file="test"):
        """Helper to handle path types using the standard filepath."""
        if dtype == "checkpoint":
            return os.path.join(self.directory, "checkpoint", base)
        elif dtype == "log":
            return os.path.join(self.directory, "log", base)
        elif dtype == "eval":
            return os.path.join(self.directory, "eval", file, base)
        else:
            raise ValueError("Invalid dtype {}.".format(dtype))

    def _path(self, dtype="checkpoints", file="test", **kwargs):
        """Get saved model file path.

        Parameters
        ----------
        dtype : str
            Path type: "eval" (evaluations), "log" (training logs),
            "checkpoint" (training saved states)
        file : str
            File name for evaluation type.
        """
        raise NotImplementedError()

    def _save_network(self, **kwargs):
        """Wrapper for ``self.learner.save_state`` with ``self._path``."""
        path = self._path(dtype="checkpoint", **kwargs)
        self.learner.checkpoint.write(path)
        print("Saved training state: {}  -->  {}".format(
            str(self.learner), path))

    def _load_network(self, **kwargs):
        """Wrapper for ``self.learner.load_state`` with ``self._path``."""
        path = self._path(dtype="checkpoint", **kwargs)
        self.learner.checkpoint.read(path).expect_partial()
        print("Loaded training state: {}  -->  {}".format(
            path, str(self.learner)))

    def _filter(self, **kwargs):
        """Get filtered view of summary dataframe."""
        df = self.summary
        for k, v in kwargs.items():
            df = df[df[k] == v]
        return df

    def _get(self, **kwargs):
        """Get item from summary dataframe."""
        try:
            return self._filter(**kwargs).iloc[0]
        except IndexError:
            raise Exception("Entry not found: {}".format(kwargs))

    def _append(
            self, train_args, training_stats, validation_stats, metadata,
            start_time):
        """Save training and validation statistics.

        Parameters
        ----------
        train_args : dict
            Training hyperparameters / arguments. Values in
            ``hyperparameter_columns`` are appended to ``summary.csv``.
        training_stats : dict
            Training statistics to append. Values in ``scalar_statistics`` are
            appended to ``summary.csv``; other values are saved in a .npz
            if present.
        validation_stats : dict
            Validation statistics; ``meta`` is saved.
        metadata : dict
            Strategy metadata; also determines saved filepath (if applicable).
        start_time : float
            Period start time.
        """
        # Save scalar summary values
        new_row = dict(
            validation=validation_stats["meta_loss"],
            time=time.time(), duration=time.time() - start_time,
            **metadata,
            **{k: train_args[k] for k in self.hyperparameter_columns},
            **{k: training_stats[k] for k in self.learner.mean_stats})
        self.summary = self.summary.append(new_row, ignore_index=True)
        self.summary.to_csv(
            os.path.join(self.directory, "summary.csv"), index=False)

        # Save other values
        if len(self.learner.stack_stats) > 0:
            data = {
                k: training_stats["__stack_" + k]
                for k in self.learner.stack_stats
            }
            dst = self._path(dtype="log", **metadata)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez(dst, **data)

    def _training_period(
            self, train_args, validation_args, metadata, eval_file=None,
            eval_args={}):
        """Run a single training period.

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
        start_time = time.time()

        # Train for ``epochs_per_period`` meta-epochs
        print("Training:")
        self.learner.network.train = True
        training_stats = self.learner.train(
            self.problems, validation=False, **train_args)

        # Compute validation loss
        print("Validating:")
        self.learner.network.train = False
        validation_stats = self.learner.train(
            self.validation_problems, validation=True,
            seed=self.validation_seed, **validation_args)

        # Save, append data, and print info
        print("imitation: {} | meta: {} | validation: {}".format(
            training_stats["imitation_loss"], training_stats["meta_loss"],
            validation_stats["meta_loss"]))
        self._save_network(**metadata)
        self._append(
            train_args, training_stats, validation_stats, metadata, start_time)

        # Evaluate (if applicable)
        if eval_file is not None:
            self.evaluate(metadata=metadata, file=eval_file, **eval_args)

    def _complete_metadata(self, metadata):
        """Complete metadata with strategy-dependent fields.

        Parameters
        ----------
        metadata : dict
            Incomplete training period metadata.

        Returns
        -------
        dict
            Dict with additional fields (or input)
        """
        return metadata

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
        metadata = self._complete_metadata(metadata)
        self._load_network(**metadata)
        self.learner.network.train = False

        results = []
        for i in range(repeat):
            opt = self.learner.network.architecture(
                self.learner.network,
                warmup=self.validation_warmup * self.validation_unroll,
                warmup_rate=self.validation_warmup_rate,
                name="OptimizerEvaluation")
            results.append(evaluate(
                opt, desc="{}/{}".format(i + 1, repeat), **kwargs))
        results = {k: np.stack([d[k] for d in results]) for k in results[0]}

        if file is not None:
            dst = self._path(dtype="eval", file=file, **metadata)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez(dst, **results)

        return results
