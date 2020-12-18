"""Loss tracking helper class."""

import tensorflow as tf
import numpy as np


class LossTracker:
    """Helper object to hold training loss."""

    def __init__(self):
        self.data = {}

    def to_numpy(self, x):
        """Convert to numpy object.

        This allows GPU memory to be freed immediately, especially if the
        tracked objects are large.
        """
        if tf.is_tensor(x):
            return x.numpy()
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, (int, float, np.number)):
            return np.array(x)
        else:
            raise TypeError(
                "LossTracker requires tensor, array, int, or float;"
                "received {} [{}] instead".format(type(x), x))

    def append(self, data):
        """Add iteration data.

        Parameters
        ----------
        data : dict
            key, value pairs to append to data dictionary.
        """
        for k, v in data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(self.to_numpy(v))

    def summarize(self, stack_stats=[], mean_stats=[]):
        """Summarize data.

        Parameters
        ----------
        stack_stats : str[]
            List of stats to reduce with np.stack.
        mean_stats : str[]
            List of stats to reduce with np.mean.

        Returns
        -------
        dict
            {key: np.stack(value)} or {key: np.mean(value)} depending on
            whether key is in stack_stats or mean_stats. Statistics can be
            reduced with both ``stack`` and ``mean``.
        """
        stack = {k: np.stack(self.data[k]) for k in stack_stats}
        mean = {k: np.mean(self.data[k]) for k in mean_stats}

        return {**stack, **mean}
