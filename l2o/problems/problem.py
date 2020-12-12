"""Base Problem Class and Problem Specification API."""

import tensorflow as tf


class Problem:
    """Training problem.

    Keyword Args
    ------------
    persistent : int
        If >0, then this indicates the number of copies of the parameters to
        hold internally so that a number of ``tf.keras.optimizers.Optimizer``
        can act on them.
        If <= 0, the problem will not own any parameters.
    noise_stddev : float
        Normally distributed noise to add to gradients during training to
        simulate minibatch noise
    distribute : None or tf.distribute.Strategy
        Distributed training tensorflow strategy.
    """

    def __init__(self, persistent=0, noise_stddev=0.0, distribute=None):

        self.noise_stddev = noise_stddev

        if distribute is None:
            distribute = tf.distribute.get_strategy()
        self.distribute = distribute

        with distribute.scope():
            if persistent:
                self.trainable_variables = [
                    [tf.Variable(v) for v in self.get_parameters()]
                    for _ in range(persistent)
                ]
                if hasattr(self, "get_internal"):
                    self.internal = self.get_internal()
            else:
                self.trainable_variables = []

    def reset(self, values=None, internal=None):
        """Reset problem.

        Keyword Args
        ------------
        values : tf.Tensor
            New value to reset parameters to. If None, parameters are reset
            using the ``get_parameters()`` method.
        internal : tf.Tensor
            New value of internal parameters. If None, internal hidden state
            is initialized using the ``get_internal()`` method.
        """
        if hasattr(self, "get_internal"):
            if internal is None:
                self.internal = self.get_internal()
            else:
                self.internal = internal

        if values is None:
            values = self.get_parameters()

        if hasattr(self, "trainable_variables"):
            for var_set in self.trainable_variables:
                for v, new in zip(var_set, values):
                    v.assign(new)

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
        return None

    def get_parameters(self, seed=None):
        """Make variables corresponding to this problem.

        Parameters
        ----------
        seed : int
            Random seed to intialize with.

        Returns
        -------
        tf.Tensor[]
            A list of tensors representing the parameters for this problem.
        """
        raise NotImplementedError()

    def get_dataset(self, seed=None, distribute=None):
        """Get problem dataset.

        Parameters
        ----------
        seed : int
            Random seed to intialize with.
        distribute : None or tf.distribute.Strategy
            Distributed training tensorflow strategy.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()


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
