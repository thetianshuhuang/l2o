
import tensorflow as tf


class Dense:

	def __init__(
			self, units, activation=tf.nn.relu,
			kernel_initializer=tf.keras.initializers.GlorotUniform,
			bias_inititializer=tf.keras.initializers.Zeros):

		
