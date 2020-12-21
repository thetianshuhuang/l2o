"""Optimizer evaluation."""

import json
import time

import tensorflow as tf
import numpy as np

from l2o.problems import load_images
from l2o import deserialize
from . import models


class EpochTimeTracker(tf.keras.callbacks.Callback):
    """Callback to add time tracking to tf.keras.Model.fit."""

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.times = []
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        self.times.append(time.time() - self.start_time)


class BatchTracker(tf.keras.callbacks.Callback):
    """Callback to track loss and accuracy on a per-batch basis."""

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.loss = []
        self.accuracy = []

    def on_train_batch_end(self, batch, logs=None):
        """Called after training each batch."""
        self.loss.append(logs.get("loss"))
        self.accuracy.append(logs.get("sparse_categorical_accuracy"))


def evaluate(
        opt, config={}, target="conv_classifier", dataset="mnist", epochs=20,
        batch_size=32):
    """Evaluate L2O.

    Parameters
    ----------
    opt : tf.keras.optimizers.Optimizer
        Optimizer to evaluate.

    Keyword Args
    ------------
    config : dict
        Problem configuration; passed to ``target``.
    target : str or callable(info, **config) -> tf.keras.Model
        Callable or name of callable in l2o.evaluate.models that creates
        model to train.
    dataset : str
        Dataset name for problems.load_images.
    epochs : int
        Number of epochs to train for
    batch_size : int
        Batch size for dataset.

    Returns
    -------
    dict
        Tensorflow "history" object.
        loss: float[]
        sparse_categorical_accuracy: float[]
        val_loss: float[]
        val_sparse_categorical_accuracy: float[]
    """
    ds_train, info_train = load_images(dataset, split="train")
    ds_val, info_val = load_images(dataset, split="test")

    model = deserialize.generic(
        target, models, pass_cond=None,
        message="training model", default="simple_conv")
    model = model(info_train, **config)

    model.compile(
        opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def _batch(ds):
        return ds.batch(
            batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    time_tracking = EpochTimeTracker()
    batch_tracking = BatchTracker()

    results = model.fit(
        _batch(ds_train.shuffle(buffer_size=batch_size * 16)),
        validation_data=_batch(ds_val),
        epochs=epochs,
        callbacks=[time_tracking, batch_tracking])

    # Add custom tracking for time, per-batch stats
    results = dict(
        **results.history, epoch_time=time_tracking.times,
        batch_loss=batch_tracking.loss, batch_accuracy=batch_tracking.accuracy)
    # numpy-ify lists
    return {k: np.array(v, dtype=np.float32) for k, v in results.items()}
