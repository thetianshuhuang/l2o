"""Custom fit method."""

import time
import tensorflow as tf
import numpy as np
from tqdm import tqdm


def model_fit(model, train, test, epochs=1, metrics=[]):
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
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, losses, axis=None)

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
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, losses, axis=None)

    # Train/test loop
    def run_loop(dataset, step):
        losses = []
        times = []
        for batch in dataset:
            losses.append(step(batch).numpy())
            times.append(time.time() - start_time)

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

    # Epoch loop
    for _ in tqdm(range(epochs)):
        train_loss, train_time, train_metrics = run_loop(train, train_step)
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

    return {k: np.array(v, dtype=np.float32) for k, v in stats.items()}
