import numpy as np
import tensorflow as tf

from tensorflow import keras
from tf.keras.layers import LSTMCell, Dense


class Clamp(keras.layers.Layer):
    def __init__(self, min_value=None, max_value=None):
        super(Clamp, self).__init__()
        self._min = min_value
        self._max = max_value

    def call(self, inputs):
        output = inputs
        if self._min is not None:
            output = tf.maximum(output, self._min)
        if self._max is not None:
            output = tf.minimum(output, self._max)
        return output


class LogAndSign(keras.layers.Layer):
    def __init__(self, k=5):
        super(LogAndSign, self).__init__()
        self._k = k

    def call(self, inputs):

        eps = np.finfo(inputs.dtype.as_numpy_dtype).eps
        ndims = inputs.get_shape().ndims

        log = tf.log(tf.abs(inputs) + eps)
        clamped_log = Clamp(min_value=-1.0)(log / self._k)
        sign = Clamp(min_value=-1.0, max_value=1.0)(inputs * np.exp(self._k))

        return tf.concat([clamped_log, sign], ndims - 1)


class CoordinateWiseLSTM:

    def __init__(self, output_size, preprocess, layers=(20, 20), **kwargs):
        defaults = {
            "kernel_initializer": "truncated_normal",
            "recurrent_initializer": "truncated_normal"
        }
        defaults.update(kwargs)

        self.preprocess = preprocess
        self.layers = [LSTMCell(hsize, **defaults) for hsize in layers]
        self.postprocess = Dense(1, input_shape=(layers[-1],))

    def call(self, inputs, states):

        input_shape = inputs.get_shape().as_list()

        x = tf.reshape(inputs, [-1, 1])
        x = self.preprocess(x)

        states_new = []
        for layer, state in zip(self.layers, states):
            x, state_new = layer(x, state)
            states_new.append(state_new)
        x = self.postprocess(x)

        return tf.reshape(x, input_shape), states_new

    def get_initial_state(self, batch_size=None):
        return [
            layer.get_initial_state(batch_size=batch_size)
            for layer in self.layers
        ]
