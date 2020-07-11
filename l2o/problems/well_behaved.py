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
    persistent : bool
        If True, then the parameters are held internally as variables to be
        used so that ``tf.keras.optimizers.Optimizer`` can act on them.
        If False, then the problem will not generate its own parameters.
    """

    def __init__(self, ndim, persistent=False, **kwargs):

        # save ndim
        self.ndim = ndim

        super().__init__(persistent=persistent)

    def size(self, unroll):
        return 1

    def get_parameters(self):
        return [tf.zeros([self.ndim, 1], tf.float32)]

    def get_internal(self):
        return {
            'W': tf.random.normal([self.ndim, self.ndim]),
            'y': tf.random.normal([self.ndim, 1])
        }

    def objective(self, params, internal):
        return tf.nn.l2_loss(
            tf.matmul(internal['W'], params[0]) - internal['y'])

    def test_objective(self, _):
        return tf.nn.l2_loss(
            tf.matmul(self.internal['W'] - self.params) - self.internal['y'])
