"""Read and interpret summary files and evaluation files."""
import collections

import pandas as pd
import numpy as np


def moving_average(a, n=10):
    """Moving average utility with truncated averages for the tails.
    
    Parameters
    ----------
    a : np.array
        Array input. Should be one dimensional.
    n : int
        Moving average window.
    """
    # Center
    cs = np.cumsum(a, dtype=float)
    cs[n:] = cs[n:] - cs[:-n]
    center = cs[n - 1:] / n

    # Tails
    left = np.cumsum(a[:int(n / 2)]) / np.arange(1, 1 + int(n / 2))
    right = np.flip(
        np.cumsum(a[-int(n / 2):][::-1]) / np.arange(1, 1 + int(n / 2)))
    return np.concatenate([left, center, right])


def plot_band(ax, x, y, label=None, color=None):
    """Plot mean and min-to-max color band for stacked data y."""
    lower, upper, mean = [f(y, axis=0) for f in np.min, np.max, np.mean]
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, lower, upper, alpha=0.2, color=color)
