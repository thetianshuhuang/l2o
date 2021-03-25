"""Plotting scripts."""

import math
from matplotlib import pyplot as plt


def _expand(base, baselines=["adam"]):
    return baselines + ["{}{}".format(base, x + 1) for x in range(4)]


def plot_training(ctx, tests, limit=False):
    """Calls plot_training (1 row per test, 4 replicas per row)."""
    vh = len(tests)
    fig, axs = plt.subplots(vh, 4, figsize=(16, 3 * vh))
    for test, row in zip(tests, axs):
        for rep, ax in enumerate(row):
            ctx.plot_training(
                "{}{}".format(test, rep + 1), ax, validation=True)
            if limit:
                ax.set_ylim(-1.1, -0.5)
    fig.tight_layout()


def plot_phase(ctx, tests, limit=False):
    """Calls plot_phase (3 columns)."""
    vh = math.ceil(len(tests) / 3)
    fig, axs = plt.subplots(vh, 3, figsize=(16, 4 * vh))
    for base, ax in zip(tests, axs.reshape(-1)):
        ctx.plot_phase(_expand(base), ax, problem="conv_train", loss=True)
        if limit:
            ax.set_xlim(-6, -2)
            ax.set_ylim(-3.5, -2.4)


def plot_phase_swarm(ctx, tests, limit=False):
    """Alias for ctx.plot_phase_swarm (3 columns)."""
    vh = math.ceil(len(tests) / 3)
    fig, axs = plt.subplots(vh, 3, figsize=(16, 4 * vh))
    for base, ax in zip(tests, axs.reshape(-1)):
        ctx.plot_phase_swarm(
            _expand(base), ax, problem="conv_train", loss=True)
        if limit:
            ax.set_xlim(-6, -2)
            ax.set_ylim(-3.5, -2.4)


def plot_stats_batch(ctx, tests, sma=100, limit=False, **kwargs):
    """Calls ctx.plot_stats_batch (2 columns)."""
    vh = math.ceil(len(tests) / 2)
    fig, axs = plt.subplots(vh, 2, figsize=(16, 4 * vh))
    for base, ax in zip(tests, axs.reshape(-1)):
        ctx.plot_stats_batch(_expand(base), ax, sma=sma, **kwargs)
        if limit:
            ax.set_ylim(-8, 0)
    fig.tight_layout()


def plot_loss(ctx, tests, **kwargs):
    """Calls ctx.plot_loss (1 row per test, val on left, train on right)."""
    vh = len(tests)
    fig, axs = plt.subplots(vh, 2, figsize=(16, 4 * vh))
    kwargs_ = {"problem": "conv_train"}
    kwargs_.update(kwargs)
    for base, row in zip(tests, axs):
        ctx.plot_loss(
            _expand(base), row[0], validation=True, **kwargs_)
        ctx.plot_loss(
            _expand(base), row[1], validation=False, **kwargs_)
    fig.tight_layout()
