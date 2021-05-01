"""Differentially Private Training."""

import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def clip_gradients_vmap(g, l2_norm_clip):
    """Clips gradients in a way that is compatible with vectorized_map.

    Taken from https://github.com/tensorflow/privacy.
    """
    grads_flat = tf.nest.flatten(g)
    squared_l2_norms = [
        tf.reduce_sum(input_tensor=tf.square(g)) for g in grads_flat]
    global_norm = tf.sqrt(tf.add_n(squared_l2_norms))
    div = tf.maximum(global_norm / l2_norm_clip, 1.)
    clipped_flat = [g / div for g in grads_flat]
    clipped_grads = tf.nest.pack_sequence_as(g, clipped_flat)
    return clipped_grads


def model_dp_fit(
        model, train, test, epochs=1, metrics=[],
        clip_norm=1.0, noise_multiplier=1.1, desc=None, debug=False):
    """Model fit fit with differential privacy.

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
    clip_norm : float
        Max l2 norm for differentially private training.
    noise_multiplier : float
        Noise magnitude.
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
        grads = tape.jacobian(loss, model.trainable_variables)

        # Apply DP
        clipped_grads = tf.vectorized_map(
            lambda g: clip_gradients_vmap(g, clip_norm), grads)

        def reduce_noise_normalize_batch(g):
            summed_gradient = tf.reduce_sum(g, axis=0)
            return summed_gradient + tf.random.normal(
                tf.shape(summed_gradient), stddev=clip_norm * noise_multiplier)

        noised_grads = [reduce_noise_normalize_batch(g) for g in clipped_grads]

        model.optimizer.apply_gradients(
            zip(noised_grads, model.trainable_variables))
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
    for _ in tqdm(range(epochs), desc=desc):
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

    res = {k: np.array(v, dtype=np.float32) for k, v in stats.items()}
    if debug:
        res.update(model.optimizer.aggregate_debug_data(trace))
    return res
