import numpy as np


def to_integer_distribution(x, name="undefined"):
    """Helper function to deserialize an unroll distribution."""
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
    """Helper function to deserialize an annealing schedule."""
    if type(x) == float:
        return lambda i: np.exp(i * -np.abs(x))
    elif type(x) in (list, tuple):
        return x.__getitem__
    elif callable(x):
        return x
    else:
        raise TypeError(
            "Unrecognized {}_schedule type; must be float, list "
            "or callable(int) -> float".format(name))


def to_integer_schedule(x, name="undefined"):
    # List -> convert to function
    if type(x) == list or type(x) == tuple:
        return x.__getitem__
    # Dict -> convert to exponential
    elif type(x) == dict:
        return lambda i: x["coefficient"] * (x["base"]**i)
    # Callable
    elif callable(x):
        return x
    else:
        raise TypeError(
            "Unrecognized {}_schedule dtype; must be int[], dict with keys "
            "'coefficient' (int), 'base' (int), or callable(int) -> int."
            "".format(name))
