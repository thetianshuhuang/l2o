"""Plotting utilities."""

import os
import json
import numpy as np
from matplotlib import pyplot as plt

from .strategy import get_constructor, Baseline


def _read_json(d):
    with open(d) as f:
        return json.load(f)


def running_mean(x, N):
    """Simple moving average."""
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def plot_band(ax, x, y, label=None, color=None, sma=0):
    """Plot mean and min-to-max color band for stacked data y."""
    if sma > 0:
        y = np.stack([running_mean(y_i, sma) for y_i in y])
        x = x[:y.shape[1]]
    lower, upper, mean = [f(y, axis=0) for f in [np.min, np.max, np.mean]]
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

    def summary(self):
        """Print summary of results."""
        for k, v in self.results.items():
            s = " [r]" if k in self._results else ""
            print("{}: {}{}".format(k, v, s))

    def _get_test(self, t):
        """Get container."""
        base = t.split(":")[0]
        if base in self.baselines:
            return Baseline(
                os.path.join(self.dir_baseline, base),
                name=self.baselines[base])
        elif base in self.results:
            if base not in self._results:
                path = os.path.join(self.dir_results, base)
                cfg = _read_json(os.path.join(path, "config.json"))
                self._results[base] = get_constructor(
                    cfg["strategy_constructor"]
                )(path, name=self.results[base])
            return self._results[base]
        else:
            raise ValueError("Unknown result: {}".format(base))

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

    def get_name(self, t, **metadata):
        """Get test full name."""
        return self._get_test(t)._display_name(**metadata)

    def get_summary(self, t, **metadata):
        """Get test summary data."""
        return self._get_test(t).get_summary(**metadata)

    _phase_args = {
        "width": 0.002, "headwidth": 5, "headlength": 5,
        "scale_units": "xy", "angles": "xy", "scale": 1
    }

    def plot_phase(
            self, tests, ax, loss=False, lgd=True, problem="conv_train"):
        """Plot test phase diagram."""
        for i, t in enumerate(tests):
            name, metadata = self._expand_name(t)
            d = self.get_eval(name, problem=problem, **metadata)
            if not loss:
                x = np.mean(d["sparse_categorical_accuracy"], axis=0)
                y = np.mean(d["val_sparse_categorical_accuracy"], axis=0)
            else:
                x = np.mean(np.log(d["loss"]), axis=0)
                y = np.mean(np.log(d["val_loss"]), axis=0)
            ax.quiver(
                x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                color='C' + str(i), **self._phase_args,
                label=self.get_name(name, **metadata))
        if loss:
            ax.set_xlabel("Log Train Loss")
            ax.set_ylabel("Log Validation Loss")
        else:
            ax.set_xlabel("Train Accuracy")
            ax.set_ylabel("Validation Accuracy")

        if lgd:
            ax.legend()

    def plot_phase_swarm(
            self, tests, ax, loss=False, lgd=True, problem="conv_train"):
        """Plot test phase diagram as a 'swarm'."""
        for i, t in enumerate(tests):
            name, metadata = self._expand_name(t)
            d = self.get_eval(name, problem=problem, **metadata)
            if not loss:
                x = d["sparse_categorical_accuracy"]
                y = d["val_sparse_categorical_accuracy"]
            else:
                x = np.log(d["loss"])
                y = np.log(d["val_loss"])

            for j, (dx, dy) in enumerate(zip(x, y)):
                if j == 0:
                    ax.quiver(
                        dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1],
                        color='C' + str(i), **self._phase_args,
                        label=self.get_name(name, **metadata))
                else:
                    ax.quiver(
                        dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1],
                        color='C' + str(i), **self._phase_args)
        if loss:
            ax.set_xlabel("Log Train Loss")
            ax.set_ylabel("Log Validation Loss")
        else:
            ax.set_xlabel("Train Accuracy")
            ax.set_ylabel("Validation Accuracy")

        if lgd:
            ax.legend()

    def plot_stats(self, tests, axs):
        """Plot test statistics."""
        for t in tests:
            name, meta = self._expand_name(t)
            d = self.get_eval(name, **meta)
            for ax, val in zip([*axs[0], *axs[1]], self._keys):
                y_val = np.log(d[val]) if val.endswith("loss") else d[val]
                plot_band(
                    ax, np.arange(25), y_val, label=self.get_name(name, **meta))

        for ax in [*axs[0], *axs[1]]:
            ax.set_xticks(np.arange(25))
            ax.set_xlabel("Epoch")

        axs[1][1].legend()

        axs[0][0].set_ylabel("Log Training Loss")
        axs[0][1].set_ylabel("Log Validation Loss")
        axs[1][0].set_ylabel("Training Accuracy")
        axs[1][1].set_ylabel("Validation Accuracy")

    def plot_loss(self, tests, ax, problem="conv_train", validation=False):
        """Plot loss curve by epoch."""
        key = "val_loss" if validation else "loss"

        for t in tests:
            name, meta = self._expand_name(t)
            d = self.get_eval(name, problem=problem, **meta)
            plot_band(
                ax, np.arange(25), np.log(d[key]),
                label=self.get_name(name, **meta))
        ax.legend()
        ax.set_ylabel("Log Val Loss" if validation else "Log Training Loss")
        ax.set_xlabel("Epochs")

    def plot_stats_batch(
            self, tests, ax, start=0, end=0, use_time=False, sma=0, loss=True,
            problem="conv_train"):
        """Plot test loss or accuracy, batch-wise."""
        for t in tests:
            name, meta = self._expand_name(t)
            d = self.get_eval(name, **meta)

            if end == 0:
                end = d["batch_loss"].shape[1]

            if use_time:
                x = np.linspace(
                    0, np.sum(d["epoch_time"]) / d["epoch_time"].shape[0],
                    num=d["batch_loss"].shape[1])[start:end]
                xlabel = "Time (s)"
            else:
                x = np.arange(start, end)
                xlabel = "Step"

            if loss:
                y = np.log(d["batch_loss"][:, start:end] + 1e-10)
            else:
                y = d["batch_accuracy"][:, start:end]

            plot_band(ax, x, y, label=self.get_name(name, **meta), sma=sma)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Log Training Loss" if loss else "Training Accuracy")
        ax.legend()

    def plot_training(self, test, ax, **kwargs):
        """Plot Meta and Imitation Loss for a single test."""
        self._get_test(test).plot_training(ax, **kwargs)