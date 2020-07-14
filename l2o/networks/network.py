import tensorflow as tf


class BaseNetwork(tf.keras.Model):

    def __init__(self, name="BaseNetwork"):

        super().__init__(name=name)

    def call(self, param, inputs, states):
        """Network call override (handled by tf.keras.Model)

        NOTE: ``inputs`` may have undefined dimensions due to gradients not
        yet being connected to the graph yet. The dimension of ``param`` should
        be used instead.

        Parameters
        ----------
        param : tf.Variable
            Corresponding input variable.
        inputs : tf.Tensor
            Inputs; should be gradients.
        states : dict
            Current hidden states; encoded by .get_initial_state

        Returns
        -------
        (tf.Tensor, dict)
            [0] : Output; gradient delta
            [1] : New state
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
