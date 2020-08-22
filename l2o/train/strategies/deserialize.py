"""Deserialization Helper Functions for Training Strategies."""
import numpy as np

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
        return [_deserialize_problem(p) for p in pset]
    else:
        return default
