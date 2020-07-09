import tensorflow as tf

from .problem import Problem


class Quadratic(Problem):
    """Simple quadratic bowl $L(x) = ||Wx-y||_2^2$

    Parameters
    ----------
    ndim : int
        Number of dimensions

    Keyword Args
    ------------
    W : tf.Tensor
        W matrix, [ndim, ndim]. If None, a random matrix is generated with
        elements from a standard normal.
    y : tf.Tensor
        y vector, [ndim]. If None, a random vector is generated with elements
        from a standard normal.
    """

    def __init__(self, ndim, W=None, y=None, **kwargs):
        # , random_seed=None, noise_stdev=0.0):

        # New or use given
        self.W = tf.Variable(
            tf.random.normal([ndim, ndim]) if W is None else W)
        self.y = tf.Variable(
            tf.random.normal([ndim, 1]) if y is None else y)

        # save ndim for clone_problem
        self.ndim = ndim

        # Always create new parameters
        self.params = tf.Variable(
            tf.zeros([ndim, 1], tf.float32), trainable=True)

        # Properties
        self.trainable_variables = [self.params]

    def size(self, unroll):
        return 1

    def clone_problem(self):
        return Quadratic(self.ndim, W=self.W, y=self.y)

    def objective(self, _):
        return tf.nn.l2_loss(tf.matmul(self.W, self.params) - self.y)

    def reset(self, copy=None):

        # New params
        self.params.assign(tf.zeros([self.ndim, 1], tf.float32))

        # New problem
        self.W.assign(tf.random.normal([self.ndim, self.ndim]))
        self.y.assign(tf.random.normal([self.ndim, 1]))

        if copy is not None:
            copy.W.assign(self.W)
            copy.y.assign(self.y)
