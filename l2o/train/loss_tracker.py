"""Loss tracking helper class."""

import tensorflow as tf
import numpy as np


class LossTracker:
    """Helper object to hold training loss.

    Parameters
    ----------
    keys : str[]
        List of data keys to track.
    """

    def __init__(self, keys):
        self.data = {k: [] for k in keys}

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
            assert(k in self.data)
            self.data[k].append(self.to_numpy(v))

    def summarize(self, use_mean=[]):
        """Summarize data.

        Parameters
        ----------
        use_mean : str[]
            List of entries where mean should be taken.

        Returns
        -------
        dict
            {key: value} or {key: np.mean(value)} depending on whether key is
            in use_mean.
        """
        def _apply(k, v):
            if k in use_mean:
                return np.mean(v) if len(v) > 0 else 0
            else:
                return v

        return {k: _apply(k, v) for k, v in self.data.items()}
