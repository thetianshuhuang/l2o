"""Outer Optimizer Training."""
import tensorflow as tf
import numpy as np
import collections

from tensorflow.keras.utils import Progbar

from .loss_tracker import LossTracker
from .utils import reset_optimizer, make_seeds, regen_optimizer_vars


MetaIteration = collections.namedtuple(
    "MetaIteration", [
        "problem", "unroll_len", "p_teacher", "validation", "seed"
    ])


class TrainingMixin:
    """Training Method Mixins for TrainableOptimizer."""

    def _meta_step(self, meta, concrete_step, data, params):
        """Helper function to run for a single step."""
        # Prepare abstract loss params (meta, imitation learning weight)
        if self.il_mode == 'switch':
            is_imitation = np.random.uniform(0, 1) < meta.p_teacher
            w_meta, w_imit = (0.0, 1.0) if is_imitation else (1.0, 0.0)
        if self.il_mode == 'sum':
            w_meta, w_imit = (1.0, meta.p_teacher)

        # Graph mode this function only
        loss, params, summary = concrete_step(
            data, params,
            meta_loss_weight=tf.constant(w_meta, dtype=tf.float32),
            imitation_loss_weight=tf.constant(w_imit, dtype=tf.float32))

        # Track loss separately depending on mode.
        if self.il_mode == 'switch' and is_imitation:
            summary["imitation"] = loss
        else:
            summary["meta"] = loss

        return loss, params, summary

    def _train(self, meta, epochs=1, repeat=1):
        """Main outer training loop.

        Parameters
        ----------
        meta : MetaIteration
            Current metaiteration parameters. See ``train`` docstring.

        Keyword Args
        ------------
        epochs : int
            Number of epochs to run for.
        repeat : int
            Number of times to repeat. Will reset at the end of every repeat.

        Returns
        -------
        float
            Mean training loss for this meta-iteration
        """
        # concrete_step, params will be assigned on first iteration.
        step = None
        params = None

        # Single progress bar
        size = meta.problem.size(meta.unroll_len)
        pbar = Progbar(size * epochs * repeat, unit_name='step')
        losses = LossTracker(self.tracked_statistics)

        # See docstring for why this is necessary
        regen_optimizer_vars(self.teachers, meta.problem.trainable_variables)

        seeds = make_seeds(meta.seed, epochs * repeat)
        for i, seed in enumerate(seeds):
            # Get new state for each repeat
            if i % epochs == 0:
                def _get_p():
                    return meta.problem.get_parameters(seed=seed)
                params = self.distribute.run(_get_p)
            # New dataset using seed for each epoch.
            dataset = meta.problem.get_dataset(meta.unroll_len, seed=seed)

            for batch in dataset:
                # State (i.e. momentum) needs to be reset
                for t in self.teachers:
                    reset_optimizer(t)
                # Only create concrete loss on first iteration
                if step is None:
                    step = self.make_concrete_step(meta, batch, params)

                # The actual step
                loss, params, summary = self._meta_step(
                    meta, step, batch, params)
                losses.append(summary)
                pbar.add(1, values=[("loss", loss)])

        return losses.summarize(use_mean=self.scalar_statistics)

    def train(
            self, problems, unroll_len=lambda: 20, p_teacher=0,
            epochs=1, repeat=1, validation=False, seed=None):
        """Run meta-training.

        Parameters
        ----------
        problems : problem.ProblemSpec[]
            List of problem specifications to build and run

        Keyword Args
        ------------
        unroll_len : Callable -> int
            Callable that returns unroll size.
        p_teacher : float
            Probability of choosing imitation learning or imitation learning
            proportional constant. Cannot be >0 if self.teachers is empty.
        epochs : int
            Number of epochs to run.
        repeat : int
            Number of repetitions to run; does not rebuild graph between runs.
        validation : bool
            If True, runs in validation mode (does not perform any parameter
            updates)
        seed : int
            Random seed to use for model initializations. If None, no specific
            seed is used. Should be set to None to reduce overfitting during
            training, but fixed during validation.

        Returns
        -------
        dict
            Dictionary containing summary data, with keys:
            "meta": float
                Mean meta loss for all problems.
            "imitation": float
                Mean imitation loss for all problems.
            k: float or np.array
                Other keys specified by self.step_callback; empty by default.
                Shape depends on whether ``k`` is in ``use_mean``.
        """
        results = LossTracker(self.tracked_statistics)
        for itr, spec in enumerate(problems):
            spec.print(itr)
            problem = spec.build(
                persistent=len(self.teachers), distribute=self.distribute)

            unroll = unroll_len()
            meta = MetaIteration(
                problem, unroll, p_teacher, validation, seed)

            results.append(self._train(meta, repeat=repeat, epochs=epochs))

        return results.summarize(use_mean=self.scalar_statistics)
