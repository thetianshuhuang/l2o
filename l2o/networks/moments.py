import tensorflow as tf


def rms_momentum(grad, m, v, beta_1=0.9, beta_2=0.9):
    """RMS Moving Geometric Average Momentum Scaling (as used by Adam)

    Notation is as described by Table 1 in scale.

    Parameters
    ----------
    grad : tf.Tensor
        Current gradient g_t
    m : tf.Tensor
        Current EMA "momentum" value m_t
    v : tf.Tensor
        Current EMA "variance" value v_t

    Keyword Args
    ------------
    beta_1 : float
        Momentum decay constant
    beta_2 : float
        Variance decay constant

    Returns
    -------
    (tf.Tensor, tf.Tensor)
        [0] : updated momentum m
        [1] : updated variance v
    """

    # At initialization (i.e. m = v = 0), initialize m and v with current grad.
    # if (tf.math.count_nonzero(m) == 0) and (tf.math.count_nonzero(v) == 0):
    #     m = grad
    #     v = tf.square(grad)
    # else:
    m = beta_1 * m + (1. - beta_1) * grad
    v = beta_2 * v + (1. - beta_2) * tf.square(grad)

    return m, v


def rms_scaling(grad, decay, ms, epsilon=1e-16):
    """RMS Moving Geometric Average Gradient Scaling (as used by RMSprop)

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
    # if tf.math.count_nonzero(ms) == 0:
    #     ms = tf.square(grad)
    # else:
    ms = (1. - decay) * (tf.square(grad)) + decay * ms

    return tf.math.asinh(grad / tf.sqrt(ms + epsilon)), ms
