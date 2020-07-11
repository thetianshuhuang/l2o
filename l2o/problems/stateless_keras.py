
import tensorflow as tf
import numpy as np


class ImagePreprocess:

    def __init__(self, scale):

        self.scale = float(scale)

    def build(self, input_shape, in_idx):
        return input_shape, in_idx

    def get_parameters(self):
        return []

    def call(self, params, x):
        return tf.cast(x, tf.float32) / self.scale


class Dense:

    def __init__(
            self, units, activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform,
            bias_inititializer=tf.keras.initializers.Zeros):

        self.kernel_initializer = kernel_initializer()
        self.bias_inititializer = bias_inititializer()

        self.units = units
        self.activation = activation

        self.built = False

    def build(self, input_shape, in_idx):

        self.input_shape = np.prod(input_shape)
        self.in_idx = in_idx
        self.out_idx = in_idx + 2
        return self.units, self.out_idx

    def get_parameters(self):
        return [
            self.kernel_initializer([self.units] + self.input_shape),
            self.bias_inititializer([self.units])
        ]

    def call(self, params, x):

        x = x.reshape([x.shape[0], [-1]])
        kernel, bias = params[self.in_idx:self.out_idx]

        if self.activation is None:
            return tf.matmul(kernel, x) + bias
        else:
            return self.activation(tf.matmul(kernel, x) + bias)


class Sequential:

    def __init__(self, layers, input_shape):

        self.layers = layers

        s = input_shape
        idx = 0
        for layer in layers:
            s, idx = layer.build(s, idx)

    def get_parameters(self):

        res = []
        for layer in self.layers:
            res += layer.get_parameters()
        return res

    def call(self, params, x):

        for layer in self.layers:
            x = layer.call(params, x)

        return x
