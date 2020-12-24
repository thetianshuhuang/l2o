"""Basic models for evaluation."""

import tensorflow as tf


def conv_classifier(
        info, activation=tf.nn.relu, layers=[[3, 16, 2], [3, 16, 2]]):
    """Convolutional classifier (identical to conv_classifier problem)."""
    layers = [
        tf.keras.layers.Conv2D(
            units, size, activation=activation,
            strides=(stride, stride))
        for size, units, stride in layers
    ]
    return tf.keras.Sequential(
        [tf.keras.layers.Input(shape=info.features['image'])] + layers + [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                info.features['label'].num_classes, activation="softmax")
        ])


def mlp_classifier(info, activation=tf.nn.relu, layers=[32]):
    """MLP classifier (identical to mlp_classifier problem)."""
    return tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=info.features['image'])]
        + [tf.keras.layers.Dense(u) for u in layers]
        + [tf.keras.layers.Dense(10, activation="softmax")]
    )


def debug_net(info, activation=tf.nn.relu):
    """Debug Network."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=info.features['image']),
        tf.keras.layers.MaxPool2D(pool_size=(4, 4)),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
