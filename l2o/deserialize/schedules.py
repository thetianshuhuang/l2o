"""Schedule deserialization."""

import numpy as np


def _err_msg(name, dtype, valid, received):
    return "{}: unrecognized {}.\nReceived: {}\nValid: {}".format(
        name, dtype, received, valid)


def _get_or_last(x):
    def f(i):
        try:
            return x[i]
        except IndexError:
            return x[-1]
    return f


def integer_distribution(x, name="undefined"):
    """Deserializes an unroll distribution."""
    if type(x) == float:
        return lambda: np.random.geometric(x)
    elif type(x) == int:
        return lambda: x
    elif callable(x):
        return x
    else:
        raise TypeError(_err_msg(
            name, "integer_distribution",
            "float, int, or callable() -> int", x))


def float_schedule(x, name="undefined"):
    """Deserializes a floating point schedule."""
    if isinstance(x, dict):
        if x["type"] == "constant":
            return lambda i: x["value"]
        elif x["type"] == "list":
            return _get_or_last(x["values"])
        elif x["type"] == "exponential":
            base = x["base"] if "base" in x else 1.0
            return lambda i: base * np.exp(i * -np.abs(x["alpha"]))
        else:
            return ValueError("Invalid float schedule: {}".format(str(x)))
    elif type(x) == float:
        return lambda i: x
    elif callable(x):
        return x
    else:
        raise TypeError(_err_msg(
            name, "float_schedule",
            "dict, list, float, or callable(int) -> float", x))


def integer_schedule(x, name="undefined"):
    """Deserializes an integer schedule."""
    if type(x) == dict:
        if x["type"] == "constant":
            return lambda i: x["value"]
        elif x["type"] == "geometric":
            return lambda i: x["coefficient"] * (x["base"]**i)
        elif x["type"] == "list":
            return _get_or_last(x["values"])
        else:
            return ValueError("Invalid integer schedule: {}".format(str(x)))
    elif type(x) == int:
        return lambda i: x
    elif callable(x):
        return x
    else:
        raise TypeError(_err_msg(
            name, "integer_schedule",
            "dict, int[], int, callable(int) -> int", x))
