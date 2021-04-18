"""Plotting utilities."""

import os
import json
import numpy as np
import functools
from matplotlib import pyplot as plt

from .strategy import get_container, Baseline, ReplicateResults
from . import plots


def _read_json(d):
    with open(d) as f:
        return json.load(f)


def running_mean(x, N):
    """Simple moving average."""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_band(ax, x, y, band_scale=0, label=None, color=None, sma=0):
    """Plot mean and min-to-max color band for stacked data y.

    Parameters
    ----------
    ax : plt.axes.Axes
        Target plot.
    x : np.array
        x data array, with shape (time).
    y : np.array
        y data array, with shape (trajectory idx, time).

    Keyword Args
    ------------
    band_scale : float
        Lower and upper band variance multiplier. If 0, uses min/max instead.
    label : str
        Line label for legend.
    color : str
        Line/shading color.
    sma : int
        Simple moving average width to be applied to data pre-averaging.
    """
    if sma > 0:
        y = np.stack([running_mean(y_i, sma) for y_i in y])
        x = x[:y.shape[1]]

    mean = np.mean(y, axis=0)

    if band_scale == 0:
        lower = np.min(y, axis=0)
        upper = np.max(y, axis=0)
    else:
        stddev = np.sqrt(np.var(y, axis=0))
        mean = np.mean(y, axis=0)
        lower = mean - band_scale * stddev
        upper = mean + band_scale * stddev

    mean_line, = ax.plot(x, mean, label=label)
    ax.fill_between(x, lower, upper, alpha=0.25, color=mean_line.get_color())
    return mean_line


class Results:
    """Results container.

    Parameters
    ----------
    results : str
        Results directory
    baseline : str
        Baseline directory
    """

    _keys = [
        "loss", "val_loss", "sparse_categorical_accuracy",
        "val_sparse_categorical_accuracy"
    ]

    def __init__(self, results="results", baseline="baseline"):

        self.dir_results = results
        self.dir_baseline = baseline

        self.baselines = {
            k: k for k in os.listdir(baseline) if os.path.isdir(k)}
        self.baselines.update(
            _read_json(os.path.join(baseline, "names.json")))

        self.results = {
            k2 + '/' + k1: k2 + '/' + k1
            for k2 in os.listdir(results)
            if os.path.isdir(os.path.join(results, k2))
            for k1 in os.listdir(os.path.join(results, k2))
        }
        self.results.update(_read_json(os.path.join(results, "names.json")))
        self._results = {}

        self._init_plots()

    def register_names(self, names):
        """Register display name aliases not already in names.json."""
        self.results.update(names)

    def summary(self):
        """Print summary of results."""
        for k, v in self.results.items():
            s = " [r]" if k in self._results else ""
            print("{}: {}{}".format(k, v, s))

    def _get_test(self, t):
        """Get container."""
        base = t.split(":")[0].split("/")
        if len(base) == 2:
            replicate = None
        else:
            replicate = "/".join(base[2:])
        base = "/".join(base[:2])

        if base in self.baselines:
            return Baseline(
                os.path.join(self.dir_baseline, base),
                name=self.baselines[base])
        elif base in self.results:
            if base not in self._results:
                path = os.path.join(self.dir_results, base)
                self._results[base] = get_container(
                    path, name=self.results[base])
            if replicate is None:
                return self._results[base]
            else:
                return self._results[base].get(replicate)
        else:
            raise ValueError("Unknown result: {}".format(base, replicate))

    def _expand_name(self, t):
        """Expand name into base and metadata."""
        if ":" in t:
            base, meta = t.split(":")
            return base, self._get_test(base)._parse_metadata(meta)
        else:
            return t, {}

    def get_eval(self, t, problem="conv_train", **metadata):
        """Get evaluation results."""
        return self._get_test(t).get_eval(problem=problem, **metadata)

    def get_eval_stats(self, t, problem="conv_train", **metadata):
        """Get evaluation statistics."""
        res = self._get_test(t).get_eval_stats(problem=problem, **metadata)

    def get_name(self, t, **metadata):
        """Get test full name."""
        return self._get_test(t)._display_name(**metadata)

    def get_summary(self, t, **metadata):
        """Get test summary data."""
        return self._get_test(t).get_summary(**metadata)

    def adjust_init_time(self, data):
        """Adjust times to ignore initialization time."""
        return plots._adjust_init_time(data)

    def _gather_eval(self, tests, problem="conv_train"):
        """Gather evaluations."""
        meta = [self._expand_name(t) for t in tests]
        data = [self.get_eval(n, problem=problem, **m) for n, m in meta]
        dnames = [self.get_name(n, **m) for n, m in meta]

        return data, dnames

    def _execute_plot(
            self, tests, ax, problem="conv_train",
            baselines=[], func=None, **kwargs):
        """Make plot."""
        if isinstance(tests, list):
            data, dnames = self._gather_eval(baselines + tests)
            func(ax, data, dnames, **kwargs)
        elif isinstance(tests, str):
            data_b, dnames_b = self._gather_eval(baselines)

            repl = self._get_test(tests)
            dnames, data = zip(*[
                (k, v.get_eval(problem=problem))
                for k, v in repl.replicates.items()
            ])

            func(ax, data_b + list(data), dnames_b + list(dnames), **kwargs)
        else:
            raise TypeError("Invalid tests type: {}".format(tests))

    def _init_plots(self):
        """Register plots."""
        for func in plots.EXPORTS:
            setattr(self, func, functools.partial(
                self._execute_plot, func=getattr(plots, func)))

    def plot_training(self, test, ax, **kwargs):
        """Plot Meta and Imitation Loss for a single test."""
        self._get_test(test).plot_training(ax, **kwargs)
