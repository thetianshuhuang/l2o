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
        Tensorflow dataset to use
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

    Attributes
    ----------
    batch_size : int
        Replica-adjusted batch size (# samples per replica)

    """

    def __init__(
            self, model, dataset, loss,
            shuffle_buffer=None, batch_size=32, size=None):

        self.dataset = dataset
        self.model = model
        self.loss = loss

        self.shuffle_buffer = shuffle_buffer
        self._size = size
        self.batch_size = batch_size

    def __adjusted_size(self, unroll):
        """Get adjusted batch size."""
        distribute = tf.distribute.get_strategy()
        return unroll * self.batch_size * distribute.num_replicas_in_sync

    def size(self, unroll):
        """Get number of batches for this unroll duration.

        Parameters
        ----------
        unroll : int
            Unroll duration

        Returns
        -------
        int
            Number of batches. Use for progress bar size.
        """
        distribute = tf.distribute.get_strategy()
        return math.floor(self._size / self.__adjusted_size(unroll))

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
        dataset = self.dataset
        if self.shuffle_buffer is not None:
            dataset = self.dataset.shuffle(self.shuffle_buffer, seed=seed)
        distribute = tf.distribute.get_strategy()
        return distribute.experimental_distribute_dataset(
            dataset
            .batch(self.__adjusted_size(unroll), drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE))

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


class ProblemSpec:
    """Simple class used for storing problem specifications.

    Parameters
    ----------
    target : target (*args, **kwargs -> problem)
        Callable used to create problem
    args : []
        Array of arguments
    kwargs : {}
        Dictionary of keyword args
    """

    def __init__(
            self, target, args, kwargs):

        self.target = target
        self.args = args
        self.kwargs = kwargs

    def build(self, *args, **kwargs):
        """Initialize this problem.

        Returns
        -------
        problem.Problem
            Class referenced by ``callable``
        """
        return self.target(*self.args, *args, **self.kwargs, **kwargs)

    def print(self, itr):
        """Print problem information."""
        print("[#{}] {}, args={}, kwargs={}".format(
            itr, self.target.__name__, self.args, self.kwargs))
