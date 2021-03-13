"""Plotting utilities."""

import numpy as np
from .base import BaseResults


class CurriculumResults(BaseResults):
    """Results container."""

    # ----- Configuration -----------------------------------------------------

    _groupby = ["stage", "period"]

    def _expand_name(self, n):
        """Extract metadata from a string name."""
        split = n.split(".") + [-1, -1, -1]
        return split[0], {
            "stage": int(split[1]),
            "period": int(split[2]),
            "repeat": int(split[3])
        }

    def _display_name(self, n, stage=-1, period=-1, repeat=-1):
        """Get display name as string."""
        if stage == -1:
            return n
        elif period == -1:
            return "{}:{}".format(n, stage)
        elif repeat == -1:
            return "{}:{}.{}".format(n, stage, period)
        else:
            return "{}:{}.{}.{}".format(n, stage, period, repeat)

    def _file_name(self, stage=0, period=0, repeat=0):
        return "stage_{}.{}.{}".format(stage, period, repeat)

    def _complete_metadata(self, t, stage=-1, period=-1, repeat=-1):
        """Complete metadata with defaults."""
        if stage == -1:
            stage = int(self.get_summary(t)["stage"].max())
        if period == -1:
            period = int(
                self.get_summary(t).iloc[
                    self.get_summary(t, stage=stage)["validation"].idxmin()
                ]["period"])
        if repeat == -1:
            repeat = int(self.get_summary(
                t, stage=stage, period=period)["repeat"].max())
        return {"stage": stage, "period": period, "repeat": repeat}

    # ----- Data Loaders ------------------------------------------------------

    # ----- Plots -------------------------------------------------------------

    def plot_training(self, test, ax, discard_rejected=True, validation=False):
        """Plot training stages."""
        ax.set_title(self.get_name(test))
        ax.set_xlabel("Training Period by Stage")
        ax.set_ylabel("Loss normalized by best loss")

        df = self.get_summary(test, discard_rejected=discard_rejected)
        stages = df["stage"].unique()

        key = "validation" if validation else "meta_loss"

        for s in stages:
            f = df[df["stage"] == s]
            best = np.abs(np.min(f[key]))
            ax.plot(
                f["period"], f[key] / best,
                label="Stage {:n} [x{:.3f}]".format(s, best))
        ax.legend()
