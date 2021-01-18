"""Outer Optimizer Training."""

import tensorflow as tf
import numpy as np
import collections

from tensorflow.keras.utils import Progbar

from .loss_tracker import LossTracker
from .utils import make_seeds
from .unroll_state import UnrollStateManager, create_state


MetaIteration = collections.namedtuple(
    "MetaIteration", [
        "problem", "unroll_len", "p_teacher", "validation", "seed"
    ])


class TrainingMixin:
    """Training Method Mixins for TrainableOptimizer."""

    @tf.function
    def _create_scaling(self, params):
        """Random Parameter Scaling.

        Described by
        "Learning Gradient Descent: Better Generalization and Longer Horizons"
        (Lv. et. al, 2017)

        Generates a random parameter-wise scale, and transforms the parameters
        with g(parameter) = f(parameter * scale)

        Parameters
        ----------
        params : tf.Tensor[]
            List of parameters to generate scale for.

        Returns
        -------
        (tf.Tensor[], tf.Tensor[])
            [0] Transformed parameters.
            [1] Parameter scales (apply multiplicatively).
        """
        spread = self.parameter_scale_spread

        def _inner(params_):
            if spread > 0.0:
                scale = [
                    tf.exp(tf.random.uniform(
                        tf.shape(p), minval=-spread, maxval=spread))
                    for p in params_]
                params_ = [p / s for p, s in zip(params_, scale)]
            else:
                scale = [tf.ones(tf.shape(p)) for p in params_]

            return params_, scale

        distribute = tf.distribute.get_strategy()
        return distribute.run(_inner, args=(params,))

    def _meta_step(self, meta, concrete_step, data, states, scale):
        """Helper function to run for a single step."""
        # Prepare abstract loss params (meta, imitation learning weight)
        if self.il_mode == 'switch':
            is_imitation = np.random.uniform(0, 1) < meta.p_teacher
            w_meta, w_imit = (0.0, 1.0) if is_imitation else (1.0, 0.0)
        if self.il_mode == 'sum':
            w_meta, w_imit = (1.0, meta.p_teacher)

        # Graph mode this function only
        params, summary = concrete_step(
            data, states, scale,
            meta_loss_weight=tf.constant(w_meta, dtype=tf.float32),
            imitation_loss_weight=tf.constant(w_imit, dtype=tf.float32))

        summary["meta_loss_weight"] = w_meta
        summary["imitation_loss_weight"] = w_imit
        return params, summary

    def _train(self, meta, depth=1, epochs=1):
        """Main outer training loop.

        Parameters
        ----------
        meta : MetaIteration
            Current metaiteration parameters. See ``train`` docstring.

        Keyword Args
        ------------
        depth : int
            Number of outer steps per outer epoch (number of outer steps
            before resetting training problem)
        epochs : int
            Number of outer epochs to run.

        Returns
        -------
        float
            Mean training loss for this meta-iteration
        """
        # concrete_step, meta state will be assigned on first iteration.
        # concrete_step is cached.
        step = meta.problem.get_step(meta)
        warmup_step = None
        policies = [self.network, *self.teachers]
        states, scale = None, None

        # Single progress bar
        size = epochs * (depth + warmup)
        pbar = Progbar(size, unit_name='step')
        losses = LossTracker()

        dataset = meta.problem.get_dataset(
            meta.unroll_len, size, seed=meta.seed)

        # NOTE: Random seeds are totally fucked
        # (I have no idea what is going on)
        # Validation may or may not be perfectly repeatable
        tf.random.set_seed(meta.seed)
        for i, batch in enumerate(dataset):
            args = (batch, states, scale)

            # Reset params & states
            if i % (depth + self.warmup) == 0:
                params = meta.problem.get_parameters(seed=meta.seed)
                params, scale = self._create_scaling(params)
                states = [create_state(p, params) for p in policies]
            # Create concrete_step; done here to capture batch shape.
            if step is None:
                step = self.make_concrete_step(meta, *args)
            if warmup_step is None and self.warmup > 0:
                warmup_step = self.make_warmup_concrete_step(meta, *args)

            # Warmup
            if i % (depth + self.warmup) < self.warmup:
                states = self.warmup_step(*args)
            # The actual step
            else:
                if meta.validation:
                    states, stats = step(*args)
                else:
                    states, stats = self._meta_step(meta, step, *args)

            losses.append(stats)
            pbar.add(1, values=[(k, stats[k]) for k in self.pbar_values])

            # Dataset size doesn't always line up
            if i >= depth * epochs:
                break

        meta.problem.save_step(step, meta)
        return losses.summarize(self.stack_stats, self.mean_stats)

    def train(
            self, problems, unroll_len=lambda: 20, p_teacher=0,
            epochs=1, depth=1, validation=False, seed=None):
        """Run meta-training.

        Parameters
        ----------
        problems : problem.Problem[]
            List of problems to build and run

        Keyword Args
        ------------
        unroll_len : Callable -> int
            Callable that returns unroll size.
        p_teacher : float
            Probability of choosing imitation learning or imitation learning
            proportional constant. Cannot be >0 if self.teachers is empty.
        depth : int
            Number of outer steps per outer epoch (number of outer steps
            before resetting training problem)
        epochs : int
            Number of outer epochs to run.
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
            "meta_loss": float
                Mean meta loss for all problems.
            "imitation_loss": float
                Mean imitation loss for all problems.
            k: float or np.array
                Other keys specified by self.step_callback; empty by default.
                Shape depends on whether ``k`` is in ``use_mean``.
        """
        results = LossTracker()
        for itr, problem in enumerate(problems):

            print("[#{}] {}".format(itr, problem.config))

            meta = MetaIteration(
                problem, unroll_len(), p_teacher, validation, seed)
            results.append(self._train(meta, depth=depth, epochs=epochs))

        return results.summarize(self.stack_stats, self.mean_stats)
