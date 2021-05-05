"""Optimizer evaluation."""

import tensorflow as tf

from l2o.problems import load_images
from l2o import deserialize
from . import models
from . import functions
from .fit import model_fit, function_fit
from .fit_dp import model_dp_fit


def evaluate_function(
        opt, config={}, target="Rastrigin",
        steps=1000, debug=False, desc=None):
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
    debug : bool
        Whether to log debug information from optimizer.get_debug_summary().

    Returns
    -------
    dict, with keys:
        loss : float[]
    """
    function = deserialize.generic(
        target, functions, pass_cond=None, message="target function")
    function = function(**config)

    return function_fit(function, opt, steps=steps, debug=debug)


def evaluate_model(
        opt, config={}, target="conv_classifier", dataset="mnist", epochs=20,
        batch_size=32, desc=None, debug=False, dp_args=None):
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
    debug : bool
        Whether to log debug information from optimizer.get_debug_summary().
    dp_args : dict or None
        Optional args for differentially private training. If None, standard
        training is used.

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
            batch_size=batch_size, drop_remainder=True
        ).prefetch(tf.data.experimental.AUTOTUNE)

    batched_train = _batch(ds_train.shuffle(
        buffer_size=batch_size * 100, reshuffle_each_iteration=True))
    batched_val = _batch(ds_val)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

    if dp_args is None:
        return model_fit(
            model, batched_train, batched_val, epochs=epochs, desc=desc,
            debug=debug, metrics=metrics)
    else:
        return model_dp_fit(
            model, batched_train, batched_val, epochs=epochs, desc=desc,
            debug=debug, metrics=metrics, **dp_args)
