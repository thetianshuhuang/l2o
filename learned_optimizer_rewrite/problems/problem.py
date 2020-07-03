import tensorflow as tf


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
        Dataset associated with this problem. Set as None if no dataset is
        defined.

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

        self.dataset = None
        self.batch_size = 1
        self.shuffle_buffer = shuffle_buffer

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

        if self.dataset is not None and self.shuffle_buffer is not None:
            self.dataset_batched = (
                self.dataset
                .shuffle(self.shuffle_buffer)
                .batch(self.batch_size * unroll))
            # Copy dataset batches
            if copy is not None:
                copy.dataset_batched = self.dataset_batched

        for param, init in zip(self.trainable_variables, self.initializers):
            param.assign(init(shape=param.shape, dtype=tf.float32))

        # Copy variable initializations
        if copy is not None:
            pairs = zip(self.trainable_variables, copy.trainable_variables)
            for param, dst in pairs:
                dst.assign(param)


class Quadratic(Problem):

    def __init__(self, ndim, w=None, y=None):
        # , random_seed=None, noise_stdev=0.0):

        # New or use given
        self.w = tf.random.normal([ndim, ndim]) if w is None else w
        self.y = tf.random.normal([ndim, 1]) if y is None else y

        # save ndim for clone_problem
        self.ndim = ndim

        # Always create new parameters
        self.params = tf.Variable(
            tf.zeros([ndim, 1], tf.float32), trainable=True)

        # Properties
        self.trainable_variables = [self.params]
        self.initializers = [tf.keras.initailizers.Zeros()]

    def clone_problem(self):
        return Quadratic(self.ndim, w=self.w, y=self.y)

    def objective(self, _):
        return tf.nn.l2_loss(tf.matmul(self.w, self.params) - self.y)


class ProblemSpec:
    def __init__(self, callable, args, kwargs):
        self.callable = callable
        self.args = args
        self.kwargs = kwargs

    def build(self):
        return self.callable(*self.args, **self.kwargs)
