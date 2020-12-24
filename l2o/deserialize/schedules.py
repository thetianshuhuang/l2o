"""Schedule deserialization."""

import numpy as np


def integer_distribution(x, name="undefined"):
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


def float_schedule(x, name="undefined"):
    """Deserializes a floating point schedule."""
    if isinstance(x, dict):
        if x["type"] == "constant":
            return lambda i: x["value"]
        elif x["type"] == "list":
            return x["values"].__getitem__
        elif x["type"] == "exponential":
            return lambda i: np.exp(i * -np.abs(x["alpha"]))
        else:
            return ValueError("Invalid float schedule: {}".format(str(x)))
    elif callable(x):
        return x
    elif type(x) == str:
        return eval(x)
    else:
        raise TypeError(
            "Unrecognized {}_schedule type; must be dict, list "
            "or callable(int) -> float".format(name))


def integer_schedule(x, name="undefined"):
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
