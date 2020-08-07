import json

import tensorflow as tf

from l2o.problems import load_images
from . import models


def evaluate(
        opt, model="simple_conv", dataset="mnist", epochs=100, batch_size=32,
        activation=tf.nn.relu):
    """Evaluate L2O.

    Parameters
    ----------
    opt : tf.keras.optimizers.Optimizer
        Optimizer to evaluate.

    Keyword Args
    ------------
    model : str or callable(info, activation) -> tf.keras.Model
        Callable or name of callable in l2o.evaluate.models that fetches model
        to train.
    dataset : str
        Dataset name for problems.load_images.
    epochs : int
        Number of epochs to train for
    batch_size : int
        Batch size for dataset.
    activation : str or callable(tf.Tensor) -> tf.Tensor
        Activation functions.

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

    if type(model) == str:
        model = getattr(models, model)
    model = model(info_train, activation=activation)

    model.compile(
        opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    def _batch(ds):
        return ds.batch(
            batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return model.fit(
        _batch(ds_train.shuffle(buffer_size=batch_size * 16)),
        validation_data=_batch(ds_val), epochs=epochs).history
