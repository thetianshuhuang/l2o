"""Plotting utilities."""

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


class CurriculumResults:
    """Results container."""

    _train_xlabel = "Period (100 meta epochs; run 4x parallel)"
    _keys = [
        "loss", "val_loss", "sparse_categorical_accuracy",
        "val_sparse_categorical_accuracy"
    ]

    def __init__(self, dir_results="results", dir_baseline="baseline"):

        self.dir_results = dir_results
        self.dir_baseline = dir_baseline

        self.baselines = {
            k: k for k in os.listdir(dir_baseline) if os.path.isdir(k)}
        self.baselines.update(
            _read_json(os.path.join(dir_baseline, "names.json")))

        self.results = {
            k2 + '/' + k1: k2 + '/' + k1
            for k2 in os.listdir(dir_results)
            if os.path.isdir(os.path.join("results", k2))
            for k1 in os.listdir(os.path.join(dir_results, k2))
        }
        self.results.update(
            _read_json(os.path.join(dir_results, "names.json")))

        self._summary_cache = {}
        self._test_cache = {}

        for k, v in self.results.items():
            print("{}: {}".format(k, v))

    def get_name(self, t, period=-1):
        """Get test full name."""
        sfx = " @ {}".format(period) if period != -1 else ""
        if t in self.baselines:
            return self.baselines[t] + sfx
        elif t in self.results:
            return self.results[t] + sfx
        else:
            return t

    def get_summary(self, t):
        """Get summary csv as DataFrame."""
        if t not in self._summary_cache:
            self._summary_cache[t] = pd.read_csv(
                os.path.join(self.dir_results, t, "summary.csv"))
        return self._summary_cache[t]

    def _get_default_period(self, t, stage=-1, period=-1, repeat=-1):
        """Get default period and repeat indices."""
        if stage != -1 and period != -1 and repeat != -1:
            return period, repeat

        s = self.get_summary(t)
        if stage == -1:
            stage = int(s["stage"].max())
        if period == -1:
            period = int(s["period"].max())
        if repeat == -1:
            repeat = int(s[s["period"] == period]["repeat"].max())
        return stage, period, repeat

    def get_eval(
            self, t, stage=-1, period=-1, repeat=-1, problem="conv_train"):
        """Get evaluation results from .npz."""
        if t in self.baselines:
            key = (t, problem)
            if key not in self._test_cache:
                self._test_cache[key] = np.load(
                    os.path.join(self.dir_baseline, t, problem + ".npz"))
            return self._test_cache[key]
        else:
            key = (t, period, repeat, problem)
            if key not in self._test_cache:
                stage, period, repeat = self._get_default_period(
                    t, stage, period, repeat)
                fn = "period_{}.{}.{}.npz".format(stage, period, repeat)
                self._test_cache[key] = np.load(
                    os.path.join(self.dir_results, t, "eval", problem, fn))
            return self._test_cache[key]

    def get_train_log(self, t, stage=-1, period=-1, repeat=-1):
        """Get log file for a single period."""
        stage, period, repeat = self._get_default_period(
            t, stage, period, repeat)
        return np.load(os.path.join(
            self.dir_results, self.results[t], "log",
            "period_{}.{}.{}.npz".format(stage, period, repeat)))

    def get_train_logs(self, t):
        """Get all log files for non-discarded training periods."""
        df = self._reject_discarded(self.get_summary(t))

        files = [
            self.get_train_log(
                t, period=int(row["period"]), repeat=int(row["repeat"]))
            for _, row in df.iterrows()
        ]
        return {
            k: np.concatenate([f[k] for f in files], axis=1)
            for k in files[0].files
        }

    def expand_name(self, n):
        """Expand name in the form of test:period:repeat."""
        n = n.split(":")
        if len(n) > 3:
            return (n[0], int(n[1]), int(n[2]), int(n[3]))
        elif len(n) == 3:
            return (n[0], int(n[1]), int(n[2]), -1)
        elif len(n) == 2:
            return (n[0], int(n[1]), -1, -1)
        else:
            return (n[0], -1, -1, -1)

    def _reject_discarded(self, df):
        return df.iloc[
            df.reset_index().groupby(df["period"])['index'].idxmax()]

    def plot_meta_loss(self, tests, ax, discard_rejected=True):
        """Plot Meta loss for multiple tests."""
        for t in tests:
            df = self.get_summary(t)
            if discard_rejected:
                df = self._reject_discarded(df)
            ax.plot(df["period"], df["meta_loss"], label=self.get_name(t))
        ax.set_xlabel(self._train_xlabel)
        ax.set_ylabel("Meta Loss")
        ax.legend()

    def plot_training(self, test, ax, discard_rejected=True, weighted=False):
        """Plot Meta and Imitation Loss for a single test."""
        ax.set_title(self.get_name(test))
        ax.set_xlabel(self._train_xlabel)

        df = self.get_summary(test)
        if discard_rejected:
            df = self._reject_discarded(df)

        if weighted:
            ax.plot(df["period"], df["meta_loss"], label="Meta Loss")        
            ax.plot(
                df["period"], df["imitation_loss"] * df["p_teacher"],
                label="Imitation Loss")
            ax.legend()
        else:
            ax.set_ylabel("Meta Loss")
            ax2 = ax.twinx()
            ax2.set_ylabel("Imitation Loss")

            a = ax.plot(
                df["period"], df["meta_loss"], label="Meta Loss", color='C0')
            b = ax2.plot(
                df["period"], df["imitation_loss"],
                label="Imitation Loss", color='C1')

            ax2.legend(a + b, [x.get_label() for x in a + b])

    def plot_training_batch(self, test, ax):
        """Plot Meta and Imitation for a single test by meta-step."""
        pass

    def plot_phase(
            self, tests, ax, loss=False, lgd=True, problem="conv_train"):
        """Plot test phase diagram."""
        for i, t in enumerate(tests):
            name, period, repeat = self.expand_name(t)
            d = self.get_eval(
                name, period=period, repeat=repeat, problem=problem)
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
                label=self.get_name(t, period=period))
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
            name, period, repeat = self.expand_name(t)
            d = self.get_eval(name, period=period, repeat=repeat)
            for ax, val in zip([*axs[0], *axs[1]], self._keys):
                y_val = np.log(d[val]) if val.endswith("loss") else d[val]
                plot_band(
                    ax, np.arange(25), y_val,
                    label=self.get_name(t, period=period))

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
            name, period, repeat = self.expand_name(t)
            d = self.get_eval(
                name, period=period, repeat=repeat, problem=problem)
            plot_band(
                ax, np.arange(25), np.log(d["loss"]),
                label=self.get_name(t, period=period))
        ax.legend()
        ax.set_ylabel("Log Training Loss")
        ax.set_xlabel("Epochs")

    def plot_stats_batch(
            self, tests, ax, start=0, end=0, use_time=False, sma=0, loss=True,
            problem="conv_train"):
        """Plot test loss or accuracy, batch-wise."""
        for t in tests:
            name, period, repeat = self.expand_name(t)
            d = self.get_eval(
                name, period=period, repeat=repeat, problem=problem)
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

            plot_band(ax, x, y, label=self.get_name(t, period=period), sma=sma)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Log Training Loss" if loss else "Training Accuracy")
        ax.legend()
