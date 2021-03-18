"""Strategy loader."""

import os
import numpy as np
import pandas as pd


def _npload(*args):
    """Helper to load .npz."""
    return np.load(os.path.join(*args) + ".npz")


class Baseline:
    """Baseline result container."""

    def __init__(self, path, name="DefaultBaseline"):
        self.directory = path
        self.name = name

    def get_eval(self, problem="conv_train", **metadata):
        """Get evaluation results from .npz."""
        return _npload(self.directory, problem)

    def _parse_metadata(self, n):
        return {}

    def _display_name(self, **metadata):
        return self.name


class BaseResult:
    """Base container."""

    _groupby = ["period"]

    def __init__(self, path, name="BaseResult"):
        self.directory = path
        self.name = name
        self.summary = pd.read_csv(os.path.join(path, "summary.csv"))

    def _display_name(self, **kwargs):
        """Get display name as string."""
        return self.name

    def _parse_metadata(self, n):
        """Extract metadata from a string name."""
        raise NotImplementedError()

    def _base_path(self, base, dtype, file="test"):
        """Helper to handle path types using the standard filepath."""
        if dtype == "checkpoint":
            return os.path.join(self.directory, "checkpoint", base)
        elif dtype == "log":
            return os.path.join(self.directory, "log", base)
        elif dtype == "eval":
            return os.path.join(self.directory, "eval", file, base)
        else:
            raise ValueError("Invalid dtype {}.".format(dtype))

    def _complete_metadata(self, metadata):
        """Complete metadata with strategy-dependent fields.

        Parameters
        ----------
        metadata : dict
            Incomplete training period metadata.

        Returns
        -------
        dict
            Dict with additional fields (or input)
        """
        return metadata

    def _path(self, dtype="checkpoints", file="test", **kwargs):
        """Get file path for saved data.

        Parameters
        ----------
        dtype : str
            Path type: "eval" (evaluations), "log" (training logs),
            "checkpoint" (training saved states)
        file : str
            File name for evaluation type.

        Returns
        -------
        str
            Absolute file path.
        """
        raise NotImplementedError()

    def get_summary(self, discard_rejected=False, **kwargs):
        """Get summary csv as DataFrame with filtering applied."""
        # Filter
        df = self.summary
        for k, v in kwargs.items():
            df = df[df[k] == v]
        # Discard
        if discard_rejected:
            return df.iloc[
                df.reset_index().groupby(
                    [df[g] for g in self._groupby]
                )['index'].idxmax()]
        return df

    def get_eval(self, problem="conv_train", **meta):
        """Get evaluation results from .npz."""
        meta = self._complete_metadata(**meta)
        return _npload(self._path(dtype="eval", file=problem, **meta))

    def get_train_log(self, **metadata):
        """Get log file for a single training period."""
        metadata = self._complete_metadata(**metadata)
        return np.load(os.path.join(
            self.dir_results, t, "log", self._file_name(**metadata)))

    def plot_training(self, ax, **kwargs):
        """Plot training summary."""
        raise NotImplementedError()


class RepeatResult(BaseResult):
    """Container for RepeatStrategy."""

    def _parse_metadata(self, n):
        """Extract metadata from a string name."""
        split = n.split(".")
        if len(split) == 1:
            return {"period": int(split[0])}
        else:
            return {"period": int(split[0]), "repeat": int(split[1])}

    def _display_name(self, period=-1, repeat=-1):
        """Get display name as string."""
        if period == -1:
            return self.name
        elif repeat == -1:
            return "{}:{}".format(self.name, period)
        else:
            return "{}:{}.{}".format(self.name, period, repeat)

    def _complete_metadata(self, period=-1, repeat=-1):
        """Complete metadata with defaults."""
        if period == -1:
            period = int(self.summary["period"].max())
        if repeat == -1:
            repeat = int(self.get_summary(period=period)["repeat"].max())
        return {"period": period, "repeat": repeat}

    def _path(self, period=0, repeat=0, dtype="checkpoint", file="test"):
        """Get file path for saved data."""
        return self._base_path(
            "period_{:n}.{:n}".format(period, repeat), dtype, file=file)

    def plot_training(self, ax, discard_rejected=True, weighted=False):
        """Training summary."""
        ax.set_title(self.name)
        ax.set_xlabel("Training Period")

        df = self.get_summary(discard_rejected=discard_rejected)

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


class CurriculumResult(BaseResult):
    """Container for CurriculumStrategy."""

    _groupby = ["stage", "period"]
    _meta_schema = ["stage", "period", "repeat"]

    def _parse_metadata(self, n):
        """Extract metadata from a string name."""
        split = n.split(".")
        if len(split) == 1:
            return {"stage": int(split[0])}
        elif len(split) == 2:
            return {"stage": int(split[0]), "period": int(split[1])}
        else:
            return {
                "stage": int(split[0]),
                "period": int(split[1]),
                "repeat": int(split[2])
            }

    def _display_name(self, stage=-1, period=-1, repeat=-1):
        """Get display name as string."""
        if stage == -1:
            return self.name
        elif period == -1:
            return "{}:{}".format(self.name, stage)
        elif repeat == -1:
            return "{}:{}.{}".format(self.name, stage, period)
        else:
            return "{}:{}.{}.{}".format(self.name, stage, period, repeat)

    def _complete_metadata(self, stage=-1, period=-1, repeat=-1):
        """Complete metadata with defaults."""
        if stage == -1:
            stage = int(self.summary["stage"].max())
        if period == -1:
            period = int(
                self.summary.iloc[
                    self.get_summary(stage=stage)["validation"].idxmin()
                ]["period"])
        if repeat == -1:
            repeat = int(
                self.get_summary(stage=stage, period=period)["repeat"].max())
        return {"stage": stage, "period": period, "repeat": repeat}

    def _path(
            self, stage=0, period=0, repeat=0,
            dtype="checkpoint", file="test"):
        """Get file path for saved data."""
        return self._base_path(
            "stage_{:n}.{:n}.{:n}".format(stage, period, repeat),
            dtype, file=file)

    def plot_training(self, ax, discard_rejected=True, validation=False):
        """Plot training summary."""
        ax.set_title(self.name)
        ax.set_xlabel("Training Period by Stage")
        ax.set_ylabel("Loss normalized by best loss")

        df = self.get_summary(discard_rejected=discard_rejected)
        stages = df["stage"].unique()

        key = "validation" if validation else "meta_loss"

        for s in stages:
            f = df[df["stage"] == s]
            best = np.abs(np.min(f[key]))
            ax.plot(
                f["period"], f[key] / best,
                label="Stage {:n} [x{:.3f}]".format(s, best))
        ax.legend()


def get_constructor(s):
    """Get result container constructor."""
    if s == "CurriculumLearningStrategy":
        return CurriculumResult
    elif s == "RepeatStrategy":
        return RepeatResult
    else:
        raise ValueError("Invalid strategy type {}".format(s))
