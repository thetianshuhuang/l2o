import tensorflow as tf
from tensorflow.keras.layers import LSTMCell, Dense


class DMOptimizer(tf.keras.Model):
    """DMOptimizer network; inherits tf.keras.Model.

    Keyword Args
    ------------
    layers : int[]
        Sizes of LSTM layers.
    name : str
        Name of optimizer network.
    **kwargs : dict
        Passed onto tf.keras.layers.LSTMCell
    """

    def __init__(self, layers=(20, 20), name="DMOptimizer", **kwargs):

        super().__init__(name=name)

        defaults = {
            "kernel_initializer": "truncated_normal",
            "recurrent_initializer": "truncated_normal"
        }
        defaults.update(kwargs)

        self.recurrent = [LSTMCell(hsize, **defaults) for hsize in layers]
        self.delta = Dense(1, input_shape=(layers[-1]))

    def call(self, param, inputs, states):
        x = tf.reshape(inputs, [-1, 1])

        states_new = {}
        for i, layer in enumerate(self.recurrent):
            hidden_name = "lstm_{}".format(i)
            x, states_new[hidden_name] = layer(x, states[hidden_name])
        x = self.delta(x)

        return tf.reshape(x, param.shape), states_new

    def get_initial_state(self, var):
        batch_size = tf.size(var)
        return {
            "lstm_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }
