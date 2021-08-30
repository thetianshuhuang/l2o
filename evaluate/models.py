"""Classification models for SGD evaluation."""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, Flatten, Dense, Activation,
    MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D)


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
            return MaxPooling2D(pool_size=(args, args))
        elif isinstance(args, list):
            f, k, s = args
            return Conv2D(f, k, activation=activation, strides=(s, s))
        else:
            raise TypeError("Not a valid layer: {}".format(args))

    if head_type == "dense":
        return Sequential(
            [Input(shape=info.features['image'].shape)]
            + [_deserialize(x) for x in layers]
            + [Flatten()]
            + [Dense(
                info.features['label'].num_classes, activation="softmax")])
    elif head_type == "average":
        return Sequential(
            [Input(shape=info.features['image'].shape)]
            + [_deserialize(x) for x in layers[:-1]]
            + [Conv2D(
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
    return Sequential(
        [Flatten(input_shape=info.features['image'].shape)]
        + [Dense(u, activation=activation) for u in layers]
        + [Dense(info.features['label'].num_classes, activation="softmax")]
    )


def nas_classifier(info, filters=16, activation="relu"):
    """Branched convolutional classifier found by NAS."""

    def conv(x):
        return Conv2D(
            filters=filters, kernel_size=(3, 3), strides=(1, 1),
            padding="same", activation=activation, use_bias=True)(x)

    def pool(x):
        return AveragePooling2D(
            pool_size=(2, 2), strides=(1, 1), padding="same")(x)

    inputs = Input(shape=info.features['image'].shape)
    node0 = conv(inputs)
    node1 = conv(node0)
    node2 = pool(node1) + conv(node0)
    node3 = node0 + conv(conv(node1)) + node2
    fc = GlobalAveragePooling2D()(node3)
    out = Dense(10, activation="softmax")(fc)

    return Model(inputs=inputs, outputs=out)


def _residual(input, width):
    init = input

    # Check if input number of filters is same as 16 * k, else create
    # convolution2d for this input
    if init.shape[-1] != width:
        init = Conv2D(width, (1, 1), activation='linear', padding='same')(init)

    x = Conv2D(width, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(width, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return init + x


def resnet(info, depth=28, width=1):
    """Resnets of varying width and depth."""
    N = (depth - 4) // 6
    inputs = Input(shape=info.features['image'].shape)

    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    for _ in range(N):
        x = _residual(x, width * 16)
    x = MaxPooling2D((2, 2))(x)
    for _ in range(N):
        x = _residual(x, width * 32)
    x = MaxPooling2D((2, 2))(x)
    for _ in range(N):
        x = _residual(x, width * 64)

    x = GlobalAveragePooling2D()(x)
    x = Dense(
        info.features['label'].num_classes, activation="softmax")(x)
    return Model(inputs, x)
