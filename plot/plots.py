"""Plotting primitives."""

import numpy as np
from matplotlib import pyplot as plt


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


def _adjust_init_time(data):
    """Adjust times to ignore initialization time."""
    if len(data.shape) == 2:
        mean_duration = np.mean(np.diff(data, axis=1))
        return data - data[:, :1] + mean_duration
    else:
        mean_duration = np.mean(np.diff(data))
        return data - data[0] + mean_duration


_phase_args = {
    "width": 0.002, "headwidth": 5, "headlength": 5,
    "scale_units": "xy", "angles": "xy", "scale": 1
}


def plot_phase(ax, data, names, loss=False, lgd=True):
    """Test phase plot."""
    for i, (d, n) in enumerate(zip(data, names)):
        if not loss:
            x = np.mean(d["sparse_categorical_accuracy"], axis=0)
            y = np.mean(d["val_sparse_categorical_accuracy"], axis=0)
        else:
            x = np.mean(np.log(d["loss"]), axis=0)
            y = np.mean(np.log(d["val_loss"]), axis=0)
        ax.quiver(
            x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
            color='C' + str(i), **_phase_args, label=n)

    if loss:
        ax.set_xlabel("Log Train Loss")
        ax.set_ylabel("Log Validation Loss")
    else:
        ax.set_xlabel("Train Accuracy")
        ax.set_ylabel("Validation Accuracy")

    if lgd:
        ax.legend()


def plot_phase_swarm(ax, data, names, loss=False, lgd=True):
    """Test phase plot as swarm."""
    for i, (d, n) in enumerate(zip(data, names)):
        if not loss:
            x = d["sparse_categorical_accuracy"]
            y = d["val_sparse_categorical_accuracy"]
        else:
            x = np.log(d["loss"])
            y = np.log(d["val_loss"])

        for j, (dx, dy) in enumerate(zip(x, y)):
            args = [dx[:-1], dy[:-1], dx[1:] - dx[:-1], dy[1:] - dy[:-1]]
            if j == 0:
                ax.quiver(*args, color='C' + str(i), **_phase_args, label=n)
            else:
                ax.quiver(*args, color='C' + str(i), **_phase_args)

    if loss:
        ax.set_xlabel("Log Train Loss")
        ax.set_ylabel("Log Validation Loss")
    else:
        ax.set_xlabel("Train Accuracy")
        ax.set_ylabel("Validation Accuracy")

    if lgd:
        ax.legend()


def plot_loss(
        ax, data, names, band_scale=0, validation=False,
        time=False, adjust_init_time=True):
    """Plot loss curve by epoch/time."""
    key = "val_loss" if validation else "loss"

    for d, n in zip(data, names):

        if time:
            x = np.mean(d["epoch_time"], axis=0)
            if adjust_init_time:
                x = _adjust_init_time(x)
        else:
            x = np.arange(d["epoch_time"].shape[1])

        plot_band(ax, x, np.log(d[key]), label=n, band_scale=band_scale)

    ax.legend()
    ax.set_ylabel("Log Val Loss" if validation else "Log Training Loss")
    ax.set_xlabel("Time (s)" if time else "Epochs")


def plot_accuracy(
        ax, data, names, band_scale=0, validation=False,
        time=False, adjust_init_time=True):
    """Plot loss curve by epoch/time."""
    key = "val_" if validation else ""

    for d, n in zip(data, names):

        if time:
            x = np.mean(d["epoch_time"], axis=0)
            if adjust_init_time:
                x = _adjust_init_time(x)
        else:
            x = np.arange(d["epoch_time"].shape[1])

        y = d[key + "sparse_categorical_accuracy"]
        plot_band(ax, x, y, label=n, band_scale=band_scale)

    ax.legend()
    ax.set_ylabel("Validation Accuracy" if validation else "Training Accuracy")
    ax.set_xlabel("Time (s)" if time else "Epochs")


def plot_stats_batch(
        ax, data, names, start=0, end=0, use_time=False, sma=0, loss=True,
        band_scale=0):
    """Plot test loss or accuracy, batch-wise."""
    for d, n in zip(data, names):

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

        plot_band(ax, x, y, label=n, sma=sma, band_scale=band_scale)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Log Training Loss" if loss else "Training Accuracy")
    ax.legend()


EXPORTS = [
    "plot_phase", "plot_phase_swarm", "plot_stats_batch",
    "plot_loss", "plot_accuracy"
]
