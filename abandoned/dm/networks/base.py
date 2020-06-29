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

# Modified:
# - split from networks file to accomodate additional networks.
# - added support for additional linear layers at the input
# - ???

"""Learning 2 Learn meta-optimizer networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import dill as pickle
import numpy as np
import six
import sonnet as snt
import tensorflow as tf

import preprocess


def save(network, sess, filename=None):
    """Save the variables contained by a network to disk."""
    to_save = collections.defaultdict(dict)
    variables = snt.get_variables_in_module(network)

    for v in variables:
        split = v.name.split(":")[0].split("/")
        module_name = split[-2]
        variable_name = split[-1]
        to_save[module_name][variable_name] = v.eval(sess)

    if filename:
        with open(filename, "wb") as f:
            pickle.dump(to_save, f)

    return to_save


@six.add_metaclass(abc.ABCMeta)
class Network(snt.RNNCore):
    """Base class for meta-optimizer networks."""

    @abc.abstractmethod
    def initial_state_for_inputs(self, inputs, **kwargs):
        """Initial state given inputs."""
        pass


def _convert_to_initializer(initializer):
    """Returns a TensorFlow initializer.

    * Corresponding TensorFlow initializer when the argument is a string (e.g.
    "zeros" -> `tf.zeros_initializer`).
    * `tf.constant_initializer` when the argument is a `numpy` `array`.
    * Identity when the argument is a TensorFlow initializer.

    Args:
        initializer: `string`, `numpy` `array` or TensorFlow initializer.

    Returns:
        TensorFlow initializer.
    """

    if isinstance(initializer, str):
        return getattr(tf, initializer + "_initializer")(dtype=tf.float32)
    elif isinstance(initializer, np.ndarray):
        return tf.constant_initializer(initializer)
    else:
        return initializer


def _get_initializers(initializers, fields):
    """Produces a nn initialization `dict` (see Linear docs for a example).

    Grabs initializers for relevant fields if the first argument is a `dict` or
    reuses the same initializer for all fields otherwise. All initializers are
    processed using `_convert_to_initializer`.

    Args:
        initializers: Initializer or <variable, initializer> dictionary.
        fields: Fields nn is expecting for module initialization.

    Returns:
        nn initialization dictionary.
    """

    result = {}
    for f in fields:
        if isinstance(initializers, dict):
            if f in initializers:
                # Variable-specific initializer.
                result[f] = _convert_to_initializer(initializers[f])
        else:
            # Common initiliazer for all variables.
            result[f] = _convert_to_initializer(initializers)

    return result


def _get_layer_initializers(initializers, layer_name, fields):
    """Produces a nn initialization dictionary for a layer.

    Calls `_get_initializers using initializers[layer_name]` if `layer_name` is
    a valid key or using initializers otherwise (reuses initializers between
    layers).

    Args:
        initializers: Initializer, <variable, initializer> dictionary,
                <layer, initializer> dictionary.
        layer_name: Layer name.
        fields: Fields nn is expecting for module initialization.

    Returns:
        nn initialization dictionary.
    """

    # No initializers specified.
    if initializers is None:
        return None

    # Layer-specific initializer.
    if isinstance(initializers, dict) and layer_name in initializers:
        return _get_initializers(initializers[layer_name], fields)

    return _get_initializers(initializers, fields)


class StandardDeepLSTM(Network):
    """LSTM layers with a Linear layer on top."""

    def __init__(
            self, output_size, layers, preprocess_name="identity",
            preprocess_options=None, scale=1.0, initializer=None,
            name="deep_lstm", tanh_output=False, num_linear_heads=1):
        """Creates an instance of `StandardDeepLSTM`.

        Args:
            output_size: Output sizes of the final linear layer.
            layers: Output sizes of LSTM layers.
            preprocess_name: Gradient preprocessing class name (in
                `l2l.preprocess` or tf modules). Default is `tf.identity`.
            preprocess_options: Gradient preprocessing options.
            scale: Gradient scaling (default is 1.0).
            initializer: Variable initializer for linear layer. See
                `snt.Linear` and `snt.LSTM` docs for more info. This parameter
                can be a string (e.g. "zeros" will be converted to
                tf.zeros_initializer).
            name: Module name.
        Modified:
            tanh_output: ???
            num_linear_heads: number of linear layers at the input
        """
        super(StandardDeepLSTM, self).__init__(name=name)

        self._output_size = output_size
        self._scale = scale

        # modified: add support for multiple linear layers
        self._num_linear_heads = num_linear_heads
        assert self._num_linear_heads >= 1
        # --------

        # modified: ???
        self.tanh_output = tanh_output
        # --------

        if hasattr(preprocess, preprocess_name):
            preprocess_class = getattr(preprocess, preprocess_name)
            self._preprocess = preprocess_class(**preprocess_options)
        else:
            self._preprocess = getattr(tf, preprocess_name)

        with tf.variable_scope(self._template.variable_scope):
            self._cores = []
            for i, size in enumerate(layers, start=1):
                name = "lstm_{}".format(i)
                init = _get_layer_initializers(
                    initializer, name, ("w_gates", "b_gates"))
                self._cores.append(
                    snt.LSTM(size, name=name, initializers=init))
            self._rnn = snt.DeepRNN(
                self._cores, skip_connections=False, name="deep_rnn")

            # modified: add support for multiple linear layers
            self._linear = [
                snt.Linear(
                    output_size, name="linear",
                    initializers=_get_layer_initializers(
                        initializer, "linear_{}".format(i), ("w", "b"))
                ) for i in range(self.num_linear_heads)
            ]
            # replaces:
            # init = _get_layer_initializers(initializer, "linear", ("w", "b"))
            # self._linear = snt.Linear(
            #     output_size, name="linear", initializers=init)
            # --------

            # modified: ???
            self.init_flag = False
            # --------

    def _build(self, inputs, prev_state, linear_i=-1):
        """Connects the `StandardDeepLSTM` module into the graph.

        Args:
            inputs: 2D `Tensor` ([batch_size, input_size]).
            prev_state: `DeepRNN` state.
        Modified:
            linear_i: ???

        Returns:
            `Tensor` shaped as `inputs`.
        """
        # Adds preprocessing dimension and preprocess.
        inputs = self._preprocess(tf.expand_dims(inputs, -1))
        # Incorporates preprocessing into data dimension.
        inputs = tf.reshape(inputs, [inputs.get_shape().as_list()[0], -1])
        output, next_state = self._rnn(inputs, prev_state)

        # modified: add support for multiple linear layers
        for layer in self._linear:
            output = layer(output)
        return output * self._scale, next_state
        # replaces:
        # return self._linear(output) * self._scale, next_state
        # --------

    def initial_state_for_inputs(self, inputs, **kwargs):
        batch_size = inputs.get_shape().as_list()[0]
        return self._rnn.initial_state(batch_size, **kwargs)
