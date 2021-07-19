"""Rewrite of standard keras layers with added perturbation."""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.ops import array_ops


class NoiseMixin:
    """Mixin to provide access to the _noise method.

    Keyword Args
    ------------
    perturbation : BasePerturbation() or None
        Perturbation type; if None, does nothing.
    """

    def __init__(self, *args, perturbation=None, **kwargs):

        self.perturbation = perturbation
        super().__init__(*args, **kwargs)

    def _noise(self, param, training=None):
        """Add perturbation."""
        if training and self.perturbation is not None:
            return self.perturbation.add(param)
        else:
            return param


class LSTMCell(NoiseMixin, tf.keras.layers.LSTMCell):
    """Keras LSTMCell equivalent."""

    def call(self, inputs, states, training=None):
        """Call override to add noise.

        Simplified version of the original ``call`` method, with dropout
        and implementation 1 removed.
        """
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        z = K.dot(inputs, self._noise(self.kernel, training))
        z += K.dot(h_tm1, self._noise(self.recurrent_kernel, training))
        if self.use_bias:
            z = K.bias_add(z, self._noise(self.bias, training))

        z = array_ops.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)

        h = o * self.activation(c)
        return h, [h, c]


class Dense(NoiseMixin, tf.keras.layers.Dense):
    """Keras Dense equivalent."""

    def call(self, inputs, training=None):
        """Call override to add noise."""
        return core_ops.dense(
            inputs,
            self._noise(self.kernel, training),
            self._noise(self.bias, training),
            self.activation,
            dtype=self._compute_dtype_object)
