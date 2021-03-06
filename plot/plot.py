"""Plotting functions."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .util import get_name, get_test


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


KEYS = [
    "loss", "val_loss",
    "sparse_categorical_accuracy",
    "val_sparse_categorical_accuracy"
]


def plot_stats(tests, axs):
    """Plot test statistics."""
    for key in tests:
        d = np.load(get_test(key))
        for ax, val in zip([*axs[0], *axs[1]], KEYS):
            y_val = np.log(d[val]) if val.endswith("loss") else d[val]
            plot_band(
                ax, np.arange(25), y_val, label=get_name(key))

    for ax in [*axs[0], *axs[1]]:
        ax.set_xticks(np.arange(25))
        ax.set_xlabel("Epoch")

    axs[1][1].legend()

    axs[0][0].set_ylabel("Log Training Loss")
    axs[0][1].set_ylabel("Log Validation Loss")
    axs[1][0].set_ylabel("Training Accuracy")
    axs[1][1].set_ylabel("Validation Accuracy")


def plot_stats_batch(
        tests, ax, end=0, use_time=False, sma=0, period=99, loss=True):
    """Plot test loss or accuracy, batch-wise."""
    for key in tests:
        d = np.load(get_test(key, period=period))

        if end == 0:
            end = d["batch_loss"].shape[1]

        if use_time:
            x = np.linspace(
                0, np.sum(d["epoch_time"]) / d["epoch_time"].shape[0],
                num=d["batch_loss"].shape[1])[:end]
            xlabel = "Time (s)"
        else:
            x = np.arange(end)
            xlabel = "Step"

        if loss:
            y = np.log(d["batch_loss"][:, :end] + 1e-4)
        else:
            y = d["batch_accuracy"][:, :end]

        plot_band(ax, x, y, label=get_name(key), sma=sma)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Log Training Loss" if loss else "Training Accuracy")
    ax.legend()


COLOR_CYCLE = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]


def plot_phase(tests, ax, loss=False):
    """Plot test phase diagram."""
    for t, c in zip(tests, COLOR_CYCLE):
        d = np.load(get_test(t))
        if not loss:
            x = np.mean(d["sparse_categorical_accuracy"], axis=0)
            y = np.mean(d["val_sparse_categorical_accuracy"], axis=0)
        else:
            x = np.log(np.mean(d["loss"], axis=0))
            y = np.log(np.mean(d["val_loss"], axis=0))
        ax.quiver(
            x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
            width=0.002, headwidth=5, headlength=5, color=c,
            scale_units="xy", angles="xy", scale=1, label=get_name(t))
    if loss:
        ax.set_xlabel("Log Train Loss")
        ax.set_ylabel("Log Validation Loss")
    else:
        ax.set_xlabel("Train Accuracy")
        ax.set_ylabel("Validation Accuracy")


TRAIN_KEYS = {
    "meta_loss": "Meta Loss",
    "imitation_loss": "Imitation Loss"
}


def plot_stats_phase(tests, axs):
    """Plot accuracy and loss phase."""
    plot_phase(tests, axs[0])
    plot_phase(tests, axs[1], loss=True)
    axs[0].legend(loc='lower right')
