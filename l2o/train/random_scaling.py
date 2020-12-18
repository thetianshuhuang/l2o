"""Random Scaling.

Described by
"Learning Gradient Descent: Better Generalization and Longer Horizons"
(Lv. et. al, 2017)
"""

import tensorflow as tf


def create_random_parameter_scaling(params, spread, seed=None):
    """Random Parameter Scaling.

    Generates a random parameter-wise scale, and transforms the parameters with
    g(parameter) = f(parameter * scale)

    Parameters
    ----------
    params : tf.Tensor[]
        List of parameters to generate scale for.
    spread : float
        Each parameter is randomly scaled by a factor sampled from a
        log uniform distribution exp(Unif([-L, L])). If the spread is 0,
        this is equivalent to a constant scale of 1.

    Keyword Args
    ------------
    seed : int or None
        Random seed to create scaling from.

    Returns
    -------
    (tf.Tensor[], callable(tf.Tensor[]) -> tf.Tensor[])
        [0] Transformed parameters.
        [1] Callable containing scale in a closure which transforms back.
    """
    if spread > 0.0:
        scale = [
            tf.exp(tf.random.uniform(
                tf.shape(p), minval=-spread, maxval=spread, seed=seed))
            for p in params]
        params = [p / s for p, s in zip(params, scale)]

        def transform(params_):
            return [p * s for p, s in zip(params_, scale)]

        return params, transform

    else:
        def transform(params_):
            return params_

        return params, transform
