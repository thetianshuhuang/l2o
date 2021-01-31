"""Warmup step."""

import time
import tensorflow as tf

from .unroll_state import UnrollStateManager, state_distance, UnrollState


class WarmupMixin:
    """Warmup Mixin."""

    def run_warmup(
            self, data, states, scale,
            unroll=20, problem=None, seed=None, warmup_rate=0.01):
        """Run Warmup.

        Parameters
        ----------
        data : tf.Tensor[]
            List of data tensors.
        states : UnrollState[]
            Initial problem parameter values and hidden state values for
            learned optimizer and teachers; created by UnrollStateManager.
        scale : tf.Tensor[]
            Random parameter scaling; applied multiplicatively.

        Keyword Args
        ------------
        unroll : int
            Number of unroll iterations
        problem : problems.Problem
            Training problem
        seed : int or None
            Seed to use for intializing parameters.
        warmup_rate : float
            SGD Learning rate during warmup period.

        Returns
        -------
        UnrollState[]
            Learner and teacher states after this unroll.
        """
        # Unbatch data
        data = [
            tf.stack(tf.split(dim, num_or_size_splits=unroll)) for dim in data]

        # Make Managers
        policy_managers = [
            UnrollStateManager(p, objective=problem.objective)
            for p in [self.network, *self.teachers]]

        params = states[0].params
        for i in tf.range(unroll):
            batch = [dim[i] for dim in data]

            # Update params with SGD
            with tf.GradientTape() as tape:
                tape.watch(params)
                objective = problem.objective(
                    [p * s for p, s in zip(params, scale)], batch)
            grads = tape.gradient(objective, params)
            params = [p - g * warmup_rate for p, g in zip(params, grads)]

            # Apply gradients to update optimizer internal states
            states = [
                mgr.apply_gradients(st, grads)
                for st, mgr in zip(states, policy_managers)]

        # Wipe states.params on return
        return [
            UnrollState(
                params=params, states=st.states, global_state=st.global_state)
            for st in states]

    @tf.function
    def warmup_step(self, data, states, scale, **kwargs):
        """Wraps warmup for parallel training.

        See ``warmup`` for docstring.
        """
        distribute = tf.distribute.get_strategy()
        return distribute.run(
            self.run_warmup, args=(data, states, scale), kwargs=kwargs)

    def make_warmup_concrete_step(self, meta, data, states, scale):
        """Get a concrete @tf.function graph for warmup_step.

        Parameters
        ----------
        meta : MetaIteration
            Namedtuple containing problem parameters.
        data : nested structure
            Sample data element for concrete function binding.
        states : UnrollState[]
            Initial problem parameter values and hidden state values for
            learned optimizer and teachers; created by UnrollStateManager.
        scale : tf.Tensor[]
            Random parameter scaling; applied multiplicatively.

        Returns
        -------
        tf.Graph
            Concrete function created with the specified problem inputs.
        """
        return self.warmup_step.get_concrete_function(
            data, states, scale, unroll=meta.unroll_len, problem=meta.problem,
            seed=meta.seed, warmup_rate=tf.constant(0., dtype=tf.float32))
