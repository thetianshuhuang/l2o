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
    test : bool
        If True, then the parameters are held internally as variables to be
        used during testing. If False, then the problem will not generate its
        own parameters.
    """

    def __init__(self, ndim, test=False, **kwargs):

        # save ndim
        self.ndim = ndim

        if test:
            self.params = tf.Variable(tf.zeros([self.ndim, 1], tf.float32))
            self.trainable_variables = [self.params]

    def size(self, unroll):
        return 1

    def get_parameters(self):
        return [tf.zeros([self.ndim, 1], tf.float32)]

    def get_internal(self):
        return tf.random.normal(
            [self.ndim, self.ndim]), tf.random.normal([self.ndim, 1])

    def objective(self, params, internal):
        W, y = internal
        return tf.nn.l2_loss(tf.matmul(W, params[0]) - y)

    def test_objective(self, _):
        return tf.nn.l2_loss(tf.matmul(self.W, self.params) - self.y)
