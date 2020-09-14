"""Deserialization Helper Functions for Training Strategies."""
import numpy as np
import tensorflow as tf

from l2o import problems


def to_integer_distribution(x, name="undefined"):
    """Deserializes an unroll distribution."""
    if type(x) == float:
        return lambda: np.random.geometric(x)
    elif type(x) == int:
        return lambda: x
    elif callable(x):
        return x
    else:
        raise TypeError(
            "Unrecognized {}_distribution type; must be float, int, or "
            "callable() -> int".format(name))


def to_float_schedule(x, name="undefined"):
    """Deserializes a floating point schedule."""
    if type(x) == float:
        return lambda i: np.exp(i * -np.abs(x))
    elif type(x) in (list, tuple):
        return x.__getitem__
    elif callable(x):
        return x
    elif type(x) == str:
        return eval(x)
    else:
        raise TypeError(
            "Unrecognized {}_schedule type; must be float, list "
            "or callable(int) -> float".format(name))


def to_integer_schedule(x, name="undefined"):
    """Deserializes an integer schedule."""
    # List -> convert to function
    if type(x) == list or type(x) == tuple:
        return x.__getitem__
    # Dict -> convert to exponential
    elif type(x) == dict:
        return lambda i: x["coefficient"] * (x["base"]**i)
    # int -> constant
    elif type(x) == int:
        return lambda i: x
    # Callable
    elif callable(x):
        return x
    # Callable in str form
    elif type(x) == str:
        return eval(x)
    else:
        raise TypeError(
            "Unrecognized {}_schedule dtype; must be int[], dict with keys "
            "'coefficient' (int), 'base' (int), or callable(int) -> int."
            "".format(name))


def deserialize_problem(p):
    """Helper function to deserialize a problem into a ProblemSpec."""
    if isinstance(p, problems.ProblemSpec):
        return p
    else:
        try:
            target = p['target']
            if type(target) == str:
                target = getattr(problems, target)
            return problems.ProblemSpec(target, p['args'], p['kwargs'])
        except Exception as e:
            raise TypeError(
                "Problem could not be deserialized: {}\n{}".format(p, e))


def deserialize_problems(pset, default=None):
    """Helper function to _deserialize_problem over a list."""
    if pset is not None:
        return [deserialize_problem(p) for p in pset]
    else:
        return default


def get_optimizer(opt):
    """Helper function to get optimizer using tf.keras.optimizers.get.

    Also includes optimizers in tensorflow_addons.optimizers if available.
    """
    # Mainline keras optimizers
    try:
        return tf.keras.optimizers.get(opt)
    # Not in tf.keras.optimizers -> try tensorflow_addons
    except ValueError as e:
        # In tensorflow_addons -> replicate tf.keras.optimizers.get behavior
        try:
            import tensorflow_addons as tfa
            if isinstance(opt, str):
                return getattr(tfa, opt)()
            elif isinstance(opt, dict):
                return getattr(tfa, opt['class_name'])(**opt['config'])
        # tensorflow_addons not available -> raise original error.
        except ModuleNotFoundError:
            print(
                "Warning: tensorflow_addons is not available. Only Keras "
                "Optimizers were searched for a match.")
            raise(e)
