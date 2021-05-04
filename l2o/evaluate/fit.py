"""Custom fit method."""

import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from tensorflow.keras.utils import Progbar


def model_fit(
        model, train, test, epochs=1, metrics=[], desc=None, debug=False):
    """Custom implementation of tf.keras.models.Model.fit.

    See https://github.com/tensorflow/tensorflow/issues/39448
    for why this is necessary (how the fuck has this not been fixed?)

    Parameters
    ----------
    model : tf.keras.models.Model
        Model to train.
    train : tf.data.Dataset
        Train dataset; should be batched and shuffled already if required.
    test : tf.data.Dataset
        Test dataset

    Keyword Args
    ------------
    epochs : int
        Number of epochs to run.
    metrics : [callable(tf.Tensor, tf.Tensor) -> float]
        List of tensorflow metrics to evaluate.
    desc : str
        Description for display.
    debug : bool
        Whether to log debug information from optimizer.get_debug_summary().
    """
    strategy = tf.distribute.get_strategy()

    # Distribute datasets to replicas
    train = strategy.experimental_distribute_dataset(train)
    test = strategy.experimental_distribute_dataset(test)

    start_time = time.time()

    # Train
    def _train_step(batch):
        x, y = batch
        with tf.GradientTape() as tape:
            y_hat = model(x, training=True)
            loss = model.compiled_loss(y, y_hat)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        for m in metrics:
            m.update_state(y, y_hat)

        return loss

    @tf.function
    def train_step(batch):
        losses = strategy.run(_train_step, args=(batch,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    # Test
    def _test_step(batch):
        x, y = batch
        y_hat = model(x, training=False)
        loss = model.compiled_loss(y, y_hat)
        for m in metrics:
            m.update_state(y, y_hat)
        return loss

    @tf.function
    def test_step(batch):
        losses = strategy.run(_test_step, args=(batch,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    # Train/test loop
    def run_loop(dataset, step, callback=None):
        losses = []
        times = []
        for batch in dataset:
            losses.append(step(batch).numpy())
            times.append(time.time() - start_time)
            if callback is not None:
                callback()

        metric_values = [m.result() for m in metrics]
        for m in metrics:
            m.reset_states()

        return losses, times, metric_values

    # List of stats to log
    stats = {
        "batch_loss": [],
        "batch_time": [],
        "loss": [],
        "val_loss": [],
        "epoch_time": [],
    }
    for m in metrics:
        stats[m.name] = []
        stats["val_" + m.name] = []

    # Debug
    if debug:
        trace = []

        def log_debug():
            trace.append(
                model.optimizer.get_debug_summary(model.trainable_variables))
    else:
        log_debug = None

    # Epoch loop
    pbar = Progbar(epochs, unit_name='epoch')
    for _ in range(epochs):
        train_loss, train_time, train_metrics = run_loop(
            train, train_step, callback=log_debug)
        stats["batch_loss"] += train_loss
        stats["batch_time"] += train_time
        stats["loss"].append(np.mean(train_loss))
        stats["epoch_time"].append(time.time() - start_time)
        for m, val in zip(metrics, train_metrics):
            stats[m.name].append(val)

        test_loss, test_time, test_metrics = run_loop(test, test_step)
        stats["val_loss"].append(np.mean(test_loss))
        for m, val in zip(metrics, test_metrics):
            stats["val_" + m.name].append(val)

        pbar.add(1, values=[
            ("train", stats["loss"][-1]), ("val", stats["val_loss"][-1])])

    res = {k: np.array(v, dtype=np.float32) for k, v in stats.items()}
    if debug:
        res.update(model.optimizer.aggregate_debug_data(trace))
    return res


def function_fit(function, optimizer, steps=1000, debug=False):
    """Fit a function using gradient descent.

    NOTE: only a single GPU is supported.

    Parameters
    ----------
    function : Object
        Should have ``loss`` method which computes the loss, and track
        parameters internally as tf.Variables which are exposed using
        a trainable_variables attribute.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer to use.

    Keyword Args
    ------------
    steps : int
        Number of gradient descent steps to perform.
    debug : bool
        Whether to log debug information from optimizer.get_debug_summary().
        Currently ignored.
    """
    strategy = tf.distribute.get_strategy()

    def _train_step():
        with tf.GradientTape() as tape:
            loss = function.loss()
        grads = tape.gradient(loss, function.trainable_variables)
        optimizer.apply_gradients(zip(grads, function.trainable_variables))
        return loss

    @tf.function
    def train_step():
        losses = strategy.run(_train_step)
        return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)

    losses = []
    for i in range(steps):
        losses.append(train_step())

    return {"loss": np.array(losses)}
