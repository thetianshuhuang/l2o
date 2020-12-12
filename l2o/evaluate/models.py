"""Basic models for evaluation."""

import tensorflow as tf


def simple_conv(info, activation=tf.nn.relu):
    """Simple convnet."""
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            32, 5, activation=activation,
            input_shape=info.features['image'].shape),
        tf.keras.layers.Conv2D(
            32, 3, strides=(2, 2), activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation="softmax")
    ])


def simple_mlp(info, activation=tf.nn.relu):
    """Simple MLP."""
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=info.features['image'].shape),
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation="softmax")
    ])


def deeper_mlp(info, activation=tf.nn.relu):
    """Deeper MLP."""
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=info.features['image'].shape),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(20, activation=activation),
        tf.keras.layers.Dense(10, activation="softmax")
    ])


def debug_net(info, activation=tf.nn.relu):
    """Debug Network."""
    return tf.keras.Sequential([
        tf.keras.layers.MaxPool2D(pool_size=(4, 4)),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
