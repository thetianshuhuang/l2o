# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Modified: split from networks.py

"""Optimizers from 'Learning to learn by gradient descent by gradient descent',
Andrychowicz et al, 2016.
"""

from .base import StandardDeepLSTM


class CoordinateWiseDeepLSTM(StandardDeepLSTM):
    """Coordinate-wise `DeepLSTM`."""

    def __init__(self, name="cw_deep_lstm", **kwargs):
        """Creates an instance of `CoordinateWiseDeepLSTM`.

        Args:
            name: Module name.
            **kwargs: Additional `DeepLSTM` args.
        """
        super(CoordinateWiseDeepLSTM, self).__init__(1, name=name, **kwargs)

    def _reshape_inputs(self, inputs):
        return tf.reshape(inputs, [-1, 1])

    def _build(self, inputs, prev_state):
        """Connects the CoordinateWiseDeepLSTM module into the graph.

        Args:
            inputs: Arbitrarily shaped `Tensor`.
            prev_state: `DeepRNN` state.

        Returns:
            `Tensor` shaped as `inputs`.
        """
        input_shape = inputs.get_shape().as_list()
        reshaped_inputs = self._reshape_inputs(inputs)

        build_fn = super(CoordinateWiseDeepLSTM, self)._build
        output, next_state = build_fn(reshaped_inputs, prev_state)

        # Recover original shape.
        return tf.reshape(output, input_shape), next_state

    def initial_state_for_inputs(self, inputs, **kwargs):
        reshaped_inputs = self._reshape_inputs(inputs)
        return super(CoordinateWiseDeepLSTM, self).initial_state_for_inputs(
                reshaped_inputs, **kwargs)


class KernelDeepLSTM(StandardDeepLSTM):
    """`DeepLSTM` for convolutional filters.

    The inputs are assumed to be shaped as convolutional filters with an extra
    preprocessing dimension ([kernel_w, kernel_h, n_input_channels,
    n_output_channels]).
    """

    def __init__(self, kernel_shape, name="kernel_deep_lstm", **kwargs):
        """Creates an instance of `KernelDeepLSTM`.

        Args:
            kernel_shape: Kernel shape (2D `tuple`).
            name: Module name.
            **kwargs: Additional `DeepLSTM` args.
        """
        self._kernel_shape = kernel_shape
        output_size = np.prod(kernel_shape)
        super(KernelDeepLSTM, self).__init__(output_size, name=name, **kwargs)

    def _reshape_inputs(self, inputs):
        transposed_inputs = tf.transpose(inputs, perm=[2, 3, 0, 1])
        return tf.reshape(transposed_inputs, [-1] + self._kernel_shape)

    def _build(self, inputs, prev_state):
        """Connects the KernelDeepLSTM module into the graph.

        Args:
            inputs: 4D `Tensor` (convolutional filter).
            prev_state: `DeepRNN` state.

        Returns:
            `Tensor` shaped as `inputs`.
        """
        input_shape = inputs.get_shape().as_list()
        reshaped_inputs = self._reshape_inputs(inputs)

        build_fn = super(KernelDeepLSTM, self)._build
        output, next_state = build_fn(reshaped_inputs, prev_state)
        transposed_output = tf.transpose(output, [1, 0])

        # Recover original shape.
        return tf.reshape(transposed_output, input_shape), next_state

    def initial_state_for_inputs(self, inputs, **kwargs):
        """Batch size given inputs."""
        reshaped_inputs = self._reshape_inputs(inputs)
        return super(KernelDeepLSTM, self).initial_state_for_inputs(
                reshaped_inputs, **kwargs)


class Sgd(Network):
    """Identity network which acts like SGD."""

    def __init__(self, learning_rate=0.001, name="sgd"):
        """Creates an instance of the Identity optimizer network.

        Args:
            learning_rate: constant learning rate to use.
            name: Module name.
        """
        super(Sgd, self).__init__(name=name)
        self._learning_rate = learning_rate

    def _build(self, inputs, _):
        return -self._learning_rate * inputs, []

    def initial_state_for_inputs(self, inputs, **kwargs):
        return []


def _update_adam_estimate(estimate, value, b):
    return (b * estimate) + ((1 - b) * value)


def _debias_adam_estimate(estimate, b, t):
    return estimate / (1 - tf.pow(b, t))


class Adam(Network):
    """Adam algorithm (https://arxiv.org/pdf/1412.6980v8.pdf)."""

    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
                             name="adam"):
        """Creates an instance of Adam."""
        super(Adam, self).__init__(name=name)
        self._learning_rate = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _build(self, g, prev_state):
        """Connects the Adam module into the graph."""
        b1 = self._beta1
        b2 = self._beta2

        g_shape = g.get_shape().as_list()
        g = tf.reshape(g, (-1, 1))

        t, m, v = prev_state

        t_next = t + 1

        m_next = _update_adam_estimate(m, g, b1)
        m_hat = _debias_adam_estimate(m_next, b1, t_next)

        v_next = _update_adam_estimate(v, tf.square(g), b2)
        v_hat = _debias_adam_estimate(v_next, b2, t_next)

        update = -self._learning_rate * m_hat / (tf.sqrt(v_hat) + self._epsilon)
        return tf.reshape(update, g_shape), (t_next, m_next, v_next)

    def initial_state_for_inputs(self, inputs, dtype=tf.float32, **kwargs):
        batch_size = int(np.prod(inputs.get_shape().as_list()))
        t = tf.zeros((), dtype=dtype)
        m = tf.zeros((batch_size, 1), dtype=dtype)
        v = tf.zeros((batch_size, 1), dtype=dtype)
        return (t, m, v)
