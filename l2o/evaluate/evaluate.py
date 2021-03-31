"""Optimizer evaluation."""

import tensorflow as tf

from l2o.problems import load_images
from l2o import deserialize
from . import models
from . import functions
from .fit import model_fit, function_fit


def evaluate_function(
        opt, config={}, target="Rastrigin", steps=1000, desc=None):
    """Evaluate L2O on a function.

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
    steps : int
        Number of gradient descent steps to perform.
    desc : float
        Evaluation description.

    Returns
    -------
    dict, with keys:
        loss : float[]
    """
    function = deserialize.generic(
        target, functions, pass_cond=None, message="target function")
    function = function(**config)

    return function_fit(function, opt, steps=steps)


def evaluate_model(
        opt, config={}, target="conv_classifier", dataset="mnist", epochs=20,
        batch_size=32, desc=None):
    """Evaluate L2O on a classifier model.

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
    desc : float
        Evaluation description.

    Returns
    -------
    dict, with keys:
        loss: float[]
        batch_loss: float[]
        sparse_categorical_accuracy: float[]
        val_loss: float[]
        val_sparse_categorical_accuracy: float[]
        batch_time : float[]
        epoch_time : float[]
    """
    ds_train, info_train = load_images(dataset, split="train")
    ds_val, info_val = load_images(dataset, split="test")

    model = deserialize.generic(
        target, models, pass_cond=None, message="training model")
    model = model(info_train, **config)
    model.compile(
        optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy())

    def _batch(ds):
        return ds.batch(
            batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return model_fit(
        model,
        _batch(ds_train.shuffle(
            buffer_size=batch_size * 100, reshuffle_each_iteration=True)),
        _batch(ds_val), epochs=epochs, desc=desc,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
