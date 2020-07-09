

class Problem:
    """Training problem."""

    # def get_dataset(self, unroll):
    #     """Prepare dataset.

    #     Parameters
    #     ----------
    #     unroll : int
    #         Number of unroll iterations. Batch should have size
    #         ``unroll * batch_size``.

    #     Returns
    #     -------
    #     tf.data.Dataset | [None]
    #         Dataset batched and shuffled as desired. Returns ``[None]`` if
    #         no dataset is associated with this problem or this problem is a
    #         full batch problem.
    #     """
    #     return [None]

    def size(self, unroll):
        """Get number of batches for this unroll duration."""
        return None

    def get_parameters(self):
        """Make variables corresponding to this problem.

        Returns
        -------
        tf.Tensor()
            A tuple of tensors representing the parameters for this problem.
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

    def build(self):
        """Initialize this problem

        Returns
        -------
        problem.Problem
            Class referenced by ``callable``
        """
        return self.callable(*self.args, **self.kwargs)

    def print(self, itr):
        print("--------- Problem #{} ---------".format(itr))
        print("{}, args={}, kwargs={}".format(
            self.callable.__name__, self.args, self.kwargs))
