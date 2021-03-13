"""Plotting utilities."""

from .base import BaseResults


class RepeatResults(BaseResults):
    """Results container."""

    # ----- Configuration -----------------------------------------------------

    _groupby = ["period"]

    def _expand_name(self, n):
        """Extract metadata from a string name."""
        split = n.split(".") + [-1, -1]
        return split[0], {"period": int(split[1]), "repeat": int(split[2])}

    def _display_name(self, n, period=-1, repeat=-1):
        """Get display name as string."""
        if period == -1:
            return n
        elif repeat == -1:
            return "{}:{}".format(n, period)
        else:
            return "{}:{}.{}".format(n, period, repeat)

    def _file_name(self, period=0, repeat=0):
        """Get file name as string."""
        return "period_{}.{}".format(period, repeat)

    def _complete_metadata(self, t, period=-1, repeat=-1):
        """Complete metadata with defaults."""
        if period == -1:
            period = int(self.get_summary(t)["period"].max())
        if repeat == -1:
            repeat = int(self.get_summary(t, period=period)["repeat"].max())
        return {"period": period, "repeat": repeat}

    # ----- Data Loaders ------------------------------------------------------

    def get_train_logs(self, t):
        """Get all log files for non-discarded training periods."""
        df = self.get_summary(t, reject_discarded=True)

        files = [
            self.get_train_log(
                t, period=int(row["period"]), repeat=int(row["repeat"]))
            for _, row in df.iterrows()
        ]
        return {
            k: np.concatenate([f[k] for f in files], axis=1)
            for k in files[0].files
        }

    # ----- Plots -------------------------------------------------------------

    def plot_meta_loss(self, tests, ax, discard_rejected=True):
        """Plot Meta loss for multiple tests."""
        for t in tests:
            df = self.get_summary(t, discard_rejected=discard_rejected)
            ax.plot(df["period"], df["meta_loss"], label=self.get_name(t))
        ax.set_xlabel("Training Period")
        ax.set_ylabel("Meta Loss")
        ax.legend()

    def plot_training(self, test, ax, discard_rejected=True, weighted=False):
        """Plot Meta and Imitation Loss for a single test."""
        ax.set_title(self.get_name(test))
        ax.set_xlabel("Training Period")

        df = self.get_summary(test, discard_rejected=discard_rejected)

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
