"""Abstracted unroll state for training loop inner optimization."""

import tensorflow as tf
import collections

UnrollState = collections.namedtuple(
    "UnrollState", ["params", "states", "global_state"])


class AlwaysTrue:
    """Placeholder iterator that always returns True."""

    def __iter__(self):
        """No action."""
        return self

    def __next__(self):
        """Always returns true. Infinite iterable."""
        return True


@tf.function
def create_state(policy, params, policy_ref=None, mask=AlwaysTrue()):
    """Abstracted unroll state for training loop inner optimization.

    Parameters
    ----------
    policy : BaseLearnToOptimizePolicy
        Policy to create state for.
    params : object
        Nested structure of tensors describing initial problems state.

    Keyword Args
    ------------
    policy_ref : BaseLearnToOptimizePolicy
        Policy to use for non-masked elements.
    mask : bool[]
        Mask indicating which parameters should be trained on, and which
        should use the reference optimizer (policy_ref).

    Returns
    -------
    UnrollState
        Created state for this policy.
    """
    def _inner(params):
        states = [
            (policy if mask else policy_ref).get_initial_state(p)
            for p, mask in zip(params, mask)]

        return UnrollState(
            params=params, states=states,
            global_state=policy.get_initial_state_global())

    distribute = tf.distribute.get_strategy()
    return distribute.run(_inner, args=[params])


class UnrollStateManager:
    """Unroll state -- sort of.

    Since tensorflow's behavior related to objects in loops is quite
    broken, UnrollState is implemented with what should be class attributes
    awkwardly stored externally. These attributes must be supplied on each
    method call.

    Parameters
    ----------
    policy : l2o.policies.BaseLearnToOptimizePolicy
        Optimization gradient policy to apply.

    Keyword Args
    ------------
    objective : callable(params, batch) -> float
        Computes objective value from current parameters and data batch.
    learner_mask : bool[]
        Mask indicating which parameters should be trained on, and which
        should use the reference optimizer (policy_ref).
    """

    def __init__(
            self, policy, objective=None, learner_mask=AlwaysTrue()):
        self.policy = policy
        self.objective = objective
        self.mask = learner_mask

    def advance_param(self, args, mask, global_state):
        """Advance a single parameter, depending on mask."""
        if mask:
            return self.policy.call(*args, global_state)
        else:
            return self.policy_ref.call(*args)

    def apply_gradients(self, unroll_state, grads):
        """Apply gradients."""
        # delta p, state <- policy(params, grads, local, global)
        dparams, states_new = list(map(list, zip(*[
            self.advance_param(args, mask, unroll_state.global_state)
            for mask, args in zip(
                self.mask,
                zip(unroll_state.params, grads, unroll_state.states))
        ])))

        # global_state <- global_policy(local states, global state)
        global_state_new = self.policy.call_global(
            [s for s, mask in zip(states_new, self.mask) if mask],
            unroll_state.global_state)
        unroll_state_new = UnrollState(
            params=dparams, states=states_new, global_state=global_state_new)

        # Catch NaN, make no changes
        nan_gradients = tf.math.reduce_any([
            tf.math.reduce_any(tf.math.is_nan(dp))
            for dp in dparams])
        if nan_gradients:
            return unroll_state
        else:
            return unroll_state_new

    def advance_state(self, unroll_state, batch, scale):
        """Advance this state by a single inner step.

        Parameters
        ----------
        unroll_state : UnrollState
            Unroll state data.
        batch : object
            Nested structure containing data batch.
        scale : tf.Tensor[]
            Parameter scaling to apply to parameters coordinatewise by
            multiplication.

        Returns
        -------
        float
            Objective value.
        """
        # 1. objective <- objective(params, batch)
        with tf.GradientTape() as tape:
            tape.watch(unroll_state.params)
            objective = self.objective(
                [p * s for p, s in zip(unroll_state.params, scale)], batch)
        # 2. grads <- gradient(objective, params)
        grads = tape.gradient(objective, unroll_state.params)
        # 3. delta p, state <- policy(params, grads, local, global)
        #    global_state <- global_policy(local states, global state)
        dstate = self.apply_gradients(unroll_state, grads)
        # 4. p <- p - delta p
        params_new = [
            p - d for p, d in zip(unroll_state.params, dstate.params)]

        return objective, UnrollState(
            params=params_new, states=dstate.states,
            global_state=dstate.global_state)


def state_distance(s1, s2):
    """Compute mean log parameter l2 distance between two states."""
    dist = tf.math.reduce_logsumexp([
        tf.math.reduce_logsumexp(
            2 * tf.math.log(tf.math.abs(self_p - ref_p)))
        for self_p, ref_p in zip(s1.params, s2.params)
    ])
    size = tf.math.reduce_sum([tf.size(p) for p in s1.params])
    return dist - tf.math.log(tf.cast(size, tf.float32))
