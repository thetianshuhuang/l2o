"""
"""

from networks.base import _get_layer_initializers
import sonnet as snt


class FullyConnected(snt.AbstractModule):

	def __init__(self, initializer, dim, name="preprocess_fc"):
		super(FullyConnected, self).__init__(name=name)
		init = _get_layer_initializers(
			initializer, "input_projection", ("w", "b"))
		self._linear = snt.Linear(
			dim, name="input_projection", initializers=init)

	def _build(self, inputs):
		return tf.nn.elu(self._linear(inputs))
