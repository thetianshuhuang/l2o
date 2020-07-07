import tensorflow as tf
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
    dataset : tf.data.Dataset
        Dataset associated with this problem. Set as [None] if no dataset is
        defined (note: must be iterable, hence [None] instead of None)

    Keyword Args
    ------------
    shuffle_buffer : int
        Shuffle buffer size for dataset shuffling (see tf.data.Dataset). If
        None, then the dataset is not shuffled. Does nothing if no dataset
        is associated with this problem.
    batch_size : int
        Batch size for dataset
    """

    def __init__(self, shuffle_buffer=None, batch_size=1):

        self.dataset = [None]
        self.batch_size = 1
        self.shuffle_buffer = shuffle_buffer

    def get_size(self, unroll):
        """Get size of dataset, batched as specified for BPTT training

        Parameters
        ----------
        unroll : int
            Unroll length

        Returns
        -------
        int
            Number of BPTT batches when split into batches with size
            ``batch_size * unroll``
        """

        try:
            return len(self.dataset)
        except TypeError:
            return (
                self.dataset
                .shuffle(self.shuffle_buffer)
                .batch(self.batch_size * unroll)
                .reduce(0, lambda x, _: x + 1))

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

    def reset(self, unroll, copy=None):
        """Reset trainable variables.

        This will be called at the beginning of each training operation. If
        keeping parameters between iterations is desired, this method can do
        nothing.

        If a dataset is present, it will also be shuffled.

        Parameters
        ----------
        unroll : int
            Unroll length. Will batch as ``unroll * batch_size``.
        copy : problem.Problem
            If not None, copy will be reset with the exact same parameters.
        """

        do_shuffle = (
            isinstance(self.dataset, tf.data.Dataset)
            and self.shuffle_buffer is not None)
        if do_shuffle:
            self.dataset_batched = (
                self.dataset
                .shuffle(self.shuffle_buffer)
                .batch(self.batch_size * unroll))
            # Copy dataset batches
            if copy is not None:
                copy.dataset_batched = self.dataset_batched

        if type(self.initializers) == list:
            pairs = zip(self.trainable_variables, self.initializers)
            for param, init in pairs:
                param.assign(init(shape=param.shape, dtype=tf.float32))
        else:
            for param in self.trainable_variables:
                param.assign(
                    self.initializers(shape=param.shape, dtype=tf.float32))

        # Copy variable initializations
        if copy is not None:
            pairs = zip(self.trainable_variables, copy.trainable_variables)
            for param, dst in pairs:
                dst.assign(param)


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

    def __init__(self, callable, args, kwargs):
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
        start = time.time()
        res = self.callable(*self.args, **self.kwargs)
        self._build_time = time.time() - start
        return res

    def print(self, itr):
        print("--------- Problem #{} ---------".format(itr))
        print("{}, {}, {}".format(
            self.callable.__name__, self.args, self.kwargs))
        print("Took {:.3f} Seconds to initialize.".format(self._build_time))
