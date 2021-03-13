"""Plotting utilities."""

import os
import os
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


class BaseResults:
    """Base container.

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
    _groupby = ["period"]

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
            if os.path.isdir(os.path.join("results", k2))
            for k1 in os.listdir(os.path.join(results, k2))
        }
        self.results.update(
            _read_json(os.path.join(results, "names.json")))

        self._summary_cache = {}
        self._test_cache = {}

        for k, v in self.results.items():
            print("{}: {}".format(k, v))

    # ----- Configuration -----------------------------------------------------

    def _expand_name(self, n):
        """Extract metadata from a string name."""
        return n, {}

    def _display_name(self, n, **metadata):
        """Get display name as string."""
        return n

    def _file_name(self, **metadata):
        """Get file name as string."""
        return "default"

    def _complete_metadata(self, t, **metadata):
        """Complete metadata with defaults."""
        return {}

    # ----- Data Loaders ------------------------------------------------------

    def get_summary(self, t, discard_rejected=False, **kwargs):
        """Get summary csv as DataFrame."""
        # Fetch
        if t not in self._summary_cache:
            self._summary_cache[t] = pd.read_csv(
                os.path.join(self.dir_results, t, "summary.csv"))
        # Filter
        df = self._summary_cache[t]
        for k, v in kwargs.items():
            df = df[df[k] == v]
        # Discard
        if discard_rejected:
            return df.iloc[
                df.reset_index().groupby(
                    [df[g] for g in self._groupby]
                )['index'].idxmax()]
        return df

    def _npload(self, *args):
        return np.load(os.path.join(*args) + ".npz")

    def get_eval(self, t, problem="conv_train", **meta):
        """Get evaluation results from .npz."""
        if t in self.baselines:
            key = (t, problem)
            if key not in self._test_cache:
                self._test_cache[key] = self._npload(
                    self.dir_baseline, t, problem)
            return self._test_cache[key]
        else:
            key = tuple([t, problem] + [meta[k] for k in sorted(meta.keys())])
            if key not in self._test_cache:
                self._test_cache[key] = self._npload(
                    self.dir_results, t, "eval", problem,
                    self._file_name(**self._complete_metadata(t, **meta)))
            return self._test_cache[key]

    def get_name(self, t, default="DefaultName", **metadata):
        """Get test full name."""
        if t in self.baselines:
            return self.baselines[t]
        elif t in self.results:
            return self._display_name(t, **metadata)
        else:
            return default

    def get_train_log(self, t, **metadata):
        """Get log file for a single training period."""
        metadata = self._complete_metadata(metadata)
        return np.load(os.path.join(
            self.dir_results, t, "log", self._file_name(**metadata)))

    # ----- Plots -------------------------------------------------------------

    def plot_training_batch(self, test, ax):
        """Plot Meta and Imitation for a single test by meta-step."""
        pass

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
                width=0.002, headwidth=5, headlength=5, color='C' + str(i),
                scale_units="xy", angles="xy", scale=1,
                label=self.get_name(name, **metadata))
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

    def plot_loss(self, tests, ax, problem="conv_train"):
        """Plot loss curve by epoch."""
        for t in tests:
            name, meta = self._expand_name(t)
            d = self.get_eval(name, problem=problem, **meta)
            plot_band(
                ax, np.arange(25), np.log(d["loss"]),
                label=self.get_name(name, **meta))
        ax.legend()
        ax.set_ylabel("Log Training Loss")
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
                y = np.log(d["batch_loss"][:, start:end] + 1e-4)
            else:
                y = d["batch_accuracy"][:, start:end]

            plot_band(ax, x, y, label=self.get_name(name, **meta), sma=sma)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Log Training Loss" if loss else "Training Accuracy")
        ax.legend()
