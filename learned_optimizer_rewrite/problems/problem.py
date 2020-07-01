import tensorflow as tf


class Problem:
    """Training problem inspired by Keras' model API.

    This class should own its parameters and state information such as
    minibatch.
    """

    def __init__(self):
        raise NotImplementedError()

    def objective(self):
        """Objective function.

        Any state information (minibatch, current parameters, etc) should be
        owned by the Problem class, and be fetched from here.
        """
        raise NotImplementedError()

    @property
    def trainable_weights(self):
        """Trainable weights.

        Should fetch the trainable_weights of any child tf.keras.Model
        objects or their equivalents.

        Returns
        -------
        tf.Variable[]
            List of trainable variables.
        """
        raise NotImplementedError()
