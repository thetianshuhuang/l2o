"""Stateless rewrite of standard keras layers.

Since tf.Variable stops gradients on assignment, it is not possible to use
standard keras layers, as keras is designed around tf.Variable. Therefore,
these layers have to be reimplemented to take their parameters as an argument
in tensor instead of variable form.
"""

import tensorflow as tf
import numpy as np
import math


class Layer:
    """Base layer class"""

    def __init__(self, name):
        self.name = name

    def build(self, input_shape, in_idx):
        """Build this layer.

        Parameters
        ----------
        input_shape : int[]
            Input tensor shape.
        in_idx : int
            Index in parameter list of this layer.

        Returns
        -------
        (int[0], int)
            [0] output shape.
            [1] index in the parameter list of the next layer.
        """
        return input_shape, in_idx

    def get_parameters(self, seed=None):
        """Get layer parameters.

        Parameters
        ----------
        seed : int
            Initializer seed.

        Returns
        -------
        tf.Tensor[]
            Created list of parameters reprenting this layer's parameters.
        """
        return []

    def call(self, params, x):
        """Call this layer.

        Parameters
        ----------
        params : tf.Tensor[]
            List of model parameters. The layer should save ``in_idx`` from
            ``build()`` and index into params.
        x : tf.Tensor
            Actual input data

        Returns
        -------
        tf.Tensor
            Layer output
        """
        raise NotImplementedError()

    def _seeded_init(self, *args, seed=None):
        """Create initializers with a given seed.

        This helper function is needed since some initializers do not accept
        seeds, which needs to be handled.
        """
        res = []
        for initializer in args:
            try:
                res.append(initializer(seed=seed))
            except TypeError:
                res.append(initializer())
        return res


class Dense(Layer):
    """Dense layer y = sigma(Wx + b)

    Parameters
    ----------
    units : int
        Dense layer size

    Keyword Args
    ------------
    activation : callable(tf.Tensor -> tf.Tensor)
        Activation function. If None, no activation is used.
    kernel_initializer : tf.keras.initializers.Initializer
        Initializer for kernel `W`
    bias_initializer : tf.keras.initializers.Initializer
        Initializer for bias `b`
    """

    def __init__(
            self, units, activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform,
            bias_initializer=tf.keras.initializers.Zeros):

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.units = units
        self.activation = activation

    def build(self, input_shape, in_idx):
        self.input_shape = np.prod(input_shape)
        self.in_idx = in_idx
        self.out_idx = in_idx + 2
        return self.units, self.out_idx

    def get_parameters(self, seed=None):
        kernel, bias = self._seeded_init(
            self.kernel_initializer, self.bias_initializer, seed=seed)
        return [kernel([self.input_shape, self.units]), bias([self.units])]

    def call(self, params, x):
        x = tf.reshape(x, [x.shape[0], -1])
        kernel, bias = params[self.in_idx:self.out_idx]

        if self.activation is None:
            return tf.matmul(kernel, x) + bias
        else:
            return self.activation(tf.matmul(x, kernel) + bias)


class Conv2D(Layer):

    def __init__(
            self, filters, kernel_size, stride=1, activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.GlorotUniform,
            bias_initializer=tf.keras.initializers.Zeros):

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation

    def build(self, input_shape, in_idx):
        self.in_idx = in_idx
        self.out_idx = in_idx + 2
        self.input_dim = input_shape[2]
        self.output_dim = [
            math.ceil(input_shape[0] / self.stride),
            math.ceil(input_shape[1] / self.stride), self.filters]
        return self.output_dim, self.out_idx

    def get_parameters(self, seed=None):
        kernel, bias = self._seeded_init(
            self.kernel_initializer, self.bias_initializer, seed=seed)
        kernel_shape = [
            self.kernel_size, self.kernel_size, self.input_dim, self.filters]
        return [kernel(kernel_shape), bias(self.output_dim)]

    def call(self, params, x):

        kernel, bias = params[self.in_idx:self.out_idx]

        # Add on filter dimension if not present
        if len(x.shape) == 3:
            x = x.reshape(list(x.shape) + [1])

        res = tf.nn.conv2d(x, kernel, [1, self.stride, self.stride, 1], "SAME")
        if self.activation is None:
            return res + bias
        else:
            return self.activation(res + bias)


class Sequential:
    """Sequential model.

    Parameters
    ----------
    layers : stateless_keras.Layer[]
        List of constituent layers
    input_shape : int[]
        Input data shape
    """

    def __init__(self, layers, input_shape):

        self.layers = layers

        s = input_shape
        idx = 0
        for layer in layers:
            s, idx = layer.build(s, idx)

    def get_parameters(self, seed=None):
        """Get model parameters

        Keyword Args
        ------------
        seed : int
            Model seed to use for initialization. If not None, the layers seeds
            are fetched from np.random.randint(0, 2**31) with the model seed as
            the random seed for the layer seeds.

        Returns
        -------
        tf.Tensor[]
            Initialized model parameters.
        """
        # Build seeds
        if seed is not None:
            np.random.seed(seed)
            seeds = [np.random.randint(0, 0x80000000) for _ in self.layers]
        else:
            seeds = [None for _ in self.layers]

        # Build layers
        res = []
        for s, layer in zip(seeds, self.layers):
            res += layer.get_parameters(seed=s)
        return res

    def call(self, params, x):
        """Call this model.

        Parameters
        ----------
        params : tf.Tensor[]
            List of model parameters. The layer should save ``in_idx`` from
            ``build()`` and index into params.
        x : tf.Tensor
            Actual input data

        Returns
        -------
        tf.Tensor
            Layer output
        """

        for layer in self.layers:
            x = layer.call(params, x)

        return x
