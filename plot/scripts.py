"""Plotting scripts."""

import math
import numpy as np
from matplotlib import pyplot as plt


def _expand(base, baselines=["adam"], sfx=""):
    return baselines + ["{}/{}{}".format(base, x + 1, sfx) for x in range(4)]


def plot_training(ctx, tests, limit=False):
    """Training summary plot (1 row per test, 4 replicas per row)."""
    vh = len(tests)
    fig, axs = plt.subplots(vh, 4, figsize=(16, 3 * vh))
    for test, row in zip(tests, axs.reshape(vh, 4)):
        for rep, ax in enumerate(row):
            ctx.plot_training(
                "{}/{}".format(test, rep + 1), ax, validation=True)
            if limit:
                ax.set_ylim(-1.1, 0.1)
    fig.tight_layout()
    return fig, axs


def plot_phase(ctx, tests, limit=False, problem="conv_train"):
    """Phase plot (3 columns)."""
    vh = math.ceil(len(tests) / 3)
    fig, axs = plt.subplots(vh, 3, figsize=(16, 4 * vh))
    for base, ax in zip(tests, axs.reshape(-1)):
        ctx.plot_phase(
            base, ax, problem=problem, baselines=["adam"], loss=True)
        if limit:
            ax.set_xlim(-6, -2)
            ax.set_ylim(-3.5, -2.4)
        ax.set_title(ctx.get_name(base))
    fig.tight_layout()
    return fig, axs


def plot_phase_swarm(ctx, tests, limit=False, problem="conv_train"):
    """Phase plot, swarm variant (3 columns)."""
    vh = math.ceil(len(tests) / 3)
    fig, axs = plt.subplots(vh, 3, figsize=(16, 4 * vh))
    for base, ax in zip(tests, axs.reshape(-1)):
        ctx.plot_phase_swarm(
            base, ax, problem=problem, baselines=["adam"], loss=True)
        if limit:
            ax.set_xlim(-6, -2)
            ax.set_ylim(-3.5, -2.4)
        ax.set_title(ctx.get_name(base))
    fig.tight_layout()
    return fig, axs


def plot_stats_batch(
        ctx, tests, sma=100, limit=False, **kwargs):
    """Batch stats (2 columns)."""
    vh = math.ceil(len(tests) / 2)
    fig, axs = plt.subplots(vh, 2, figsize=(16, 4 * vh))
    for base, ax in zip(tests, axs.reshape(-1)):
        ctx.plot_stats_batch(base, ax, baselines=["adam"], sma=sma, **kwargs)
        if limit:
            ax.set_ylim(-8, 0)
        ax.set_title(ctx.get_name(base))
    fig.tight_layout()
    return fig, axs


def plot_loss(ctx, tests, rulers=[], sfx="", **kwargs):
    """Loss plot (1 row per test, val on left, train on right)."""
    vh = len(tests)
    fig, axs = plt.subplots(vh, 2, figsize=(16, 4 * vh))
    for base, row in zip(tests, axs.reshape(vh, 2)):
        ctx.plot_loss(
            base, row[0], baselines=["adam"], validation=True, **kwargs)
        ctx.plot_loss(
            base, row[1], baselines=["adam"], validation=False, **kwargs)
        for r in rulers:
            row[0].axline(r, color='black')
        row[0].set_title(ctx.get_name(base))
        row[1].set_title(ctx.get_name(base))
    fig.tight_layout()
    return fig, axs


def boxplot(ctx, tests, **kwargs):
    """Box plot of training stats."""
    fig, axs = plt.subplots(
        1, len(tests), figsize=(min(3 * len(tests), 16), 3), sharey=True)
    for test, ax, in zip(tests, axs):
        ctx._get_test(test).boxplot(ax, **kwargs)
    fig.tight_layout()
    return fig, axs
