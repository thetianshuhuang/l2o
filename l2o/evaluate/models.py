"""Basic models for evaluation."""

import tensorflow as tf


class AveragePoolingAll(tf.keras.layers.Layer):
    """Average Pooling over all coordinates for a convolutional network."""

    def build(self, input_shape):
        """No weights to build."""
        pass

    def call(self, inputs):
        """Keras layer call."""
        return tf.math.reduce_mean(inputs, axis=(1, 2))


def conv_classifier(
        info, activation=tf.nn.relu, layers=[[16, 3, 1], 2, [32, 5, 1], 2],
        head_type="dense"):
    """Convolutional classifier (identical to conv_classifier problem)."""

    def _deserialize(args):
        if isinstance(args, int):
            return tf.keras.layers.MaxPooling2D(pool_size=(args, args))
        elif isinstance(args, list):
            f, k, s = args
            return tf.keras.layers.Conv2D(
                f, k, activation=activation, strides=(s, s))
        else:
            raise TypeError("Not a valid layer: {}".format(args))

    if head_type == "dense":
        return tf.keras.Sequential(
            [tf.keras.layers.Input(shape=info.features['image'].shape)]
            + [_deserialize(x) for x in layers]
            + [tf.keras.layers.Flatten()]
            + [tf.keras.layers.Dense(
                info.features['label'].num_classes, activation="softmax")])
    elif head_type == "average":
        return tf.keras.Sequential(
            [tf.keras.layers.Input(shape=info.features['image'].shape)]
            + [_deserialize(x) for x in layers[:-1]]
            + [tf.keras.layers.Conv2D(
                info.features["label"].num_classes, layers[-1][1],
                strides=(layers[-1][2], layers[-1][2]), activation="softmax")]
            + [AveragePoolingAll()]
        )
    else:
        raise ValueError(
            "Invalid classification head type {}. "
            "Must be 'dense' or 'average'.".format(head_type))


def mlp_classifier(info, activation=tf.nn.relu, layers=[32]):
    """MLP classifier (identical to mlp_classifier problem)."""
    return tf.keras.Sequential(
        [tf.keras.layers.Flatten(input_shape=info.features['image'].shape)]
        + [tf.keras.layers.Dense(u, activation=activation) for u in layers]
        + [tf.keras.layers.Dense(10, activation="softmax")]
    )
