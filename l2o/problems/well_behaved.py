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

        # New or use given
        self.W = tf.random.normal([ndim, ndim]) if W is None else W
        self.y = tf.random.normal([ndim, 1]) if y is None else y

        # save ndim for clone_problem
        self.ndim = ndim

    def size(self, unroll):
        return 1

    def get_parameters(self):
        return tf.zeros([self.ndim, 1], tf.float32)

    def objective(self, params, _):
        return tf.nn.l2_loss(tf.matmul(self.W, params) - self.y)
