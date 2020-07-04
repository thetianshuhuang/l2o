import tensorflow as tf


def rms_scaling(grad, decay, ms, epsilon=1e-16):
    """RMS Moving Geometric Average Scaling (as used by RMSprop & Adam)

    Parameters
    ----------
    grad : tf.Tensor
        New gradient tensor
    decay : tf.Tensor
        Decay coefficient. Tensor or scalar.
    ms : tf.Tensor
        Current mean square value

    Keyword Args
    ------------
    epsilon : float
        Epsilon used in normalization denominator

    Returns
    -------
    (tf.Tensor, tf.Tensor)
        [0] Scaled gradient
        [1] New mean square value
    """

    # At initialization (i.e. ms = 0), don't use decay and just fully commit
    # grad_vec to ms.
    if tf.math.count_nonzero(ms) == 0:
        ms = tf.square(grad)
    else:
        ms = (1. - decay) * (tf.square(grad)) + decay * ms

    return tf.math.asinh(grad / tf.sqrt(ms + epsilon)), ms
