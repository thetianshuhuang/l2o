import tensorflow as tf
from tf.keras.layers import LSTMCell, Dense


class DMOptimizer(tf.keras.Model):

    def __init__(
            self, output_size, layers=(20, 20),
            name="DMOptimizer", **kwargs):

        super(DMOptimizer, self).__init__(name=name)

        defaults = {
            "kernel_initializer": "truncated_normal",
            "recurrent_initializer": "truncated_normal"
        }
        defaults.update(kwargs)

        self.layers = [LSTMCell(hsize, **defaults) for hsize in layers]
        self.postprocess = Dense(1, input_shape=(layers[-1],))

    def call(self, inputs, states):

        input_shape = inputs.get_shape().as_list()

        x = tf.reshape(inputs, [-1, 1])

        for i, layer in enumerate(self.layers):
            hidden_name = "lstm_{}".format(i)
            x, states[hidden_name] = layer(x, states[hidden_name])
        x = self.postprocess(x)

        return tf.reshape(x, input_shape), states

    def get_initial_state(self, var):
        batch_size = tf.size(var)
        return {
            "lstm_{}".format(i): layer.get_initial_state(batch_size=batch_size)
            for i, layer in enumerate(self.layers)
        }
