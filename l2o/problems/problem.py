"""Base Problem Class and Problem Specification API."""

import math
import tensorflow as tf


class Problem:
    """Training problem.

    Parameters
    ----------
    model : object
        Core model (i.e. classifier or regression). Must have
        ``get_parameters`` and ``call`` methods.
    dataset : tf.data.Dataset
        Dataset to use.
    loss : callable (tf.Tensor, tf.Tensor -> tf.Tensor)
        Computes loss from model output and ground truth. Dataset should have
        input as the first array and output as the second.

    Keyword Args
    ------------
    shuffle_buffer : int
        Shuffle buffer size for dataset shuffling (see tf.data.Dataset). If
        None, then the dataset is not shuffled. Does nothing if no dataset
        is associated with this problem.
    batch_size : int
        Batch size for dataset
    size : int
        Number of elements in this dataset, if known.
    config : dict
        Configuration dictionary (for tracking and info).

    Attributes
    ----------
    batch_size : int
        Replica-adjusted batch size (# samples per replica)
    step : dict[(unroll, validation)] -> tf.Graph
        Concrete step for outer training; cached here per-problem due to long
        build times (>60s).
    """

    def __init__(
            self, model, dataset, loss,
            shuffle_buffer=None, batch_size=32, size=None, config=None):

        self.dataset = dataset
        
        self.model = model
        self.loss = loss

        self.shuffle_buffer = shuffle_buffer
        self._size = size
        self.batch_size = batch_size

        self.config = config

        self.step = {}

    def get_step(self, meta):
        """Get concrete step for given metaiteration settings.

        Parameters
        ----------
        meta : MetaIteration
            Namedtuple; indexes with (unroll_len, validation).
        """
        try:
            return self.step[(meta.unroll_len, meta.validation)]
        except KeyError:
            return None

    def save_step(self, step, meta):
        """Save concrete step.

        Parameters
        ----------
        step : tf.Graph
            Concrete function for this problem.
        meta : MetaIteration
            Namedtuple; indexes with (unroll_len, validation).        
        """
        self.step[(meta.unroll_len, meta.validation)] = step

    @tf.function
    def _get_parameters(self, distribute, seed=None):
        """Inner tf.function wrapping distribute.run."""
        def _inner():
            return self.model.get_parameters(seed=seed)
        return distribute.run(_inner)

    def get_parameters(self, seed=None):
        """Make variables corresponding to this problem.

        Keyword Args
        ------------
        seed : int
            Random seed to intialize with.

        Returns
        -------
        tf.Tensor[]
            A list of tensors representing the parameters for this problem.
        """
        distribute = tf.distribute.get_strategy()
        return self._get_parameters(distribute, seed=seed)

    def get_dataset(self, unroll, seed=None):
        """Get problem dataset.

        Parameters
        ----------
        unroll : int
            Unroll length (to set batch size).

        Keyword Args
        ------------
        seed : int
            Random seed to intialize with.
        """
        distribute = tf.distribute.get_strategy()
        ds = distribute.experimental_distribute_dataset(
            self.dataset.batch(self._size, drop_remainder=True))

        for batch in ds:
            return batch

    def get_batch(self, data, idx):
        """Get slice of dataset."""
        start = idx * self.batch_size
        end = start + self.batch_size
        return [dim[start:end] for dim in data]

    def objective(self, parameters, data):
        """Objective function.

        Any state information (minibatch, current parameters, etc) should be
        owned by the Problem class, and be fetched from here.

        Parameters
        ----------
        parameters : object
            Parameters for this problem; originally created by get_parameters.
        data : object
            Input data. Can be ignored if this problem does not require a
            dataset.

        Returns
        -------
        tf.Tensor
            Objective function value.
        """
        x, y = data
        return tf.reduce_mean(self.loss(y, self.model.call(parameters, x)))
