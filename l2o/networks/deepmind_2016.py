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
        Passed onto tf.keras.Model
    """

    def __init__(
            self, layers=(20, 20), name="DMOptimizer", **kwargs):

        super(DMOptimizer, self).__init__(name=name)

        defaults = {
            "kernel_initializer": "truncated_normal",
            "recurrent_initializer": "truncated_normal"
        }
        defaults.update(kwargs)

        self.recurrent = [LSTMCell(hsize, **defaults) for hsize in layers]
        self.postprocess = Dense(1, input_shape=(layers[-1],))

    def call(self, inputs, states):
        """Network call override (handled by tf.keras.Model)

        Parameters
        ----------
        inputs : tf.Tensor
            Inputs; should be gradients
        states : tf.Tensor
            Current hidden states; encoded by .get_initial_state

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            [0] : Output; gradient delta
            [1] : New state
        """

        input_shape = inputs.shape

        x = tf.reshape(inputs, [-1, 1])

        for i, layer in enumerate(self.recurrent):
            hidden_name = "lstm_{}".format(i)
            x, states[hidden_name] = layer(x, states[hidden_name])
        x = self.postprocess(x)

        return tf.reshape(x, input_shape), states

    def get_initial_state(self, var):
        """Get initial model state as a dictionary

        Parameters
        ----------
        var : tf.Variable
            Variable to create initial states for

        Returns
        -------
        dict (str -> tf.Tensor)
            Hidden state information serialized by string keys.
        """

        batch_size = tf.size(var)
        return {
            "lstm_{}".format(i): layer.get_initial_state(
                batch_size=batch_size, dtype=tf.float32)
            for i, layer in enumerate(self.recurrent)
        }
