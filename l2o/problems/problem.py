import time


class Problem:
    """Training problem inspired by Keras' model API.

    This class should own its parameters and state information such as
    minibatch.

    Attributes
    ----------
    trainable_variables : tf.Variable[]
        List of trainable variables (i.e. parameters) for this problem; Should
        be linked to the trainable_variables of any child tf.keras.Model
        objects or their equivalents.
    initializers : tf.keras.initializers.Initializer
        Initializer to use for trainable_variables.
    """

    def get_dataset(self, unroll):
        """Prepare dataset.

        Parameters
        ----------
        unroll : int
            Number of unroll iterations. Batch should have size
            ``unroll * batch_size``.

        Returns
        -------
        tf.data.Dataset | [None]
            Dataset batched and shuffled as desired. Returns ``[None]`` if
            no dataset is associated with this problem or this problem is a
            full batch problem.
        """
        return [None]

    def objective(self, data):
        """Objective function.

        Any state information (minibatch, current parameters, etc) should be
        owned by the Problem class, and be fetched from here.

        Parameters
        ----------
        data : tf.Tensor
            Input data. Can be ignored if this problem does not require a
            dataset.
        """
        raise NotImplementedError()

    def clone_problem(self):
        """Create a clone of this problem.

        This is used for imitation learning to provide copies of the problem
        for the student and teacher to interact with separately.

        Problems wrapping Keras models can use tf.keras.models.clone_model.
        """
        raise NotImplementedError()

    def sync(self, copy):
        """Set this problem's parameters to the values held by a copy.

        Use this to sync teacher and student copies between batches in a single
        problem.

        Parameters
        ----------
        copy : problem.Problem
            Copy of this problem to sync with. This problem's parameters are
            overwritten.
        """
        pairs = zip(self.trainable_variables, copy.trainable_variables)
        for var, cpy in pairs:
            var.assign(cpy)

    def reset(self, copy=None):
        """Optionally reset between epochs."""
        pass


class ProblemSpec:
    """Simple class used for storing problem specifications.

    Attributes
    ----------
    _build_time : float
        How long it took to build the most recent problem. -1 if not built.

    Parameters
    ----------
    callable : callable (*args, **kwargs -> problem)
        Callable used to create problem
    args : []
        Array of arguments
    kwargs : {}
        Dictionary of keyword args
    """

    def __init__(self, callable, args, kwargs):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs
        self._build_time = -1.

    def build(self):
        """Initialize this problem

        Returns
        -------
        problem.Problem
            Class referenced by ``callable``
        """
        start = time.time()
        res = self.callable(*self.args, **self.kwargs)
        self._build_time = time.time() - start
        return res

    def print(self, itr):
        print("--------- Problem #{} ---------".format(itr))
        print("{}, args={}, kwargs={}".format(
            self.callable.__name__, self.args, self.kwargs))
        print("Took {:.3f} Seconds to initialize.".format(self._build_time))
