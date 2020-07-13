import tensorflow as tf


class Problem:
    """Training problem.

    Keyword Args
    ------------
    persistent : bool
        If True, then the parameters are held internally as variables to be
        used so that ``tf.keras.optimizers.Optimizer`` can act on them.
        If False, then the problem will not generate its own parameters.
    noise_stddev : float
        Normally distributed noise to add to gradients during training to
        simulate minibatch noise
    """

    def __init__(self, persistent=False, noise_stddev=0.0):

        self.noise_stddev = noise_stddev

        if persistent:
            self.trainable_variables = [
                tf.Variable(v) for v in self.get_parameters()]
            if hasattr(self, "get_internal"):
                self.internal = self.get_internal()

    def reset(self, values=None):
        """Reset problem.

        Keyword Args
        ------------
        values : tf.Tensor
            New value to reset parameters to. If None, parameters are reset
            using the ``get_parameters()`` method.
        """
        if hasattr(self, "get_internal"):
            self.internal = self.get_internal()

        if values is None:
            values = self.get_parameters()

        for v, new in zip(self.trainable_variables, values):
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

    def get_parameters(self):
        """Make variables corresponding to this problem.

        Returns
        -------
        tf.Tensor[]
            A list of tensors representing the parameters for this problem.
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
    callable : callable (*args, **kwargs -> problem)
        Callable used to create problem
    args : []
        Array of arguments
    kwargs : {}
        Dictionary of keyword args
    """

    def __init__(
            self, callable, args, kwargs):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def build(self, *args, **kwargs):
        """Initialize this problem

        Returns
        -------
        problem.Problem
            Class referenced by ``callable``
        """
        return self.callable(*self.args, *args, **self.kwargs, **kwargs)

    def print(self, itr):
        print("--------- Problem #{} ---------".format(itr))
        print("{}, args={}, kwargs={}".format(
            self.callable.__name__, self.args, self.kwargs))
