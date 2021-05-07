"""Plotting and analysis functions."""

from .container import Results, plot_band
from .scripts import (
    plot_training, plot_phase, plot_stats_batch, plot_loss, plot_phase_swarm,
    boxplot)
from . import plots

__all__ = [
    "plots",
    "plot_band",
    "Results",
    "plot_training",
    "plot_phase",
    "plot_stats_batch",
    "plot_loss",
    "plot_phase_swarm",
    "boxplot"
]
