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

    def __get_stack(self, k):
        """Helper to get stacked key."""
        try:
            return self.data["__stack_" + k]
        except KeyError:
            return self.data[k]

    def summarize(self, stack_stats=[], mean_stats=[]):
        """Summarize data.

        NOTE
        ----
        (1) Keys for stack_stats are mutated to allow the same key to appear in
            both stack_stats and mean_stats.
        (2) When stacking a value that appears in both stack_stats and
            mean_stats, the already-stacked value is preferred if present.

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
        stack = {
            "__stack_" + k: np.stack(self.__get_stack(k)) for k in stack_stats}
        mean = {k: np.mean(self.data[k]) for k in mean_stats}

        return {**stack, **mean}
