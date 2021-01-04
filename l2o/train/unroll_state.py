"""Abstracted unroll state for training loop inner optimization."""

import tensorflow as tf
import collections

UnrollState = collections.namedtuple(
    "UnrollState", ["params", "states", "global_state"])


class AlwaysTrue:
    def __iter__(self):
        return self
    def __next__(self):
        return True


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
    get_objective : callable(params, batch) -> float
        Computes objective value from current parameters and data batch.
    transform : callable(tf.Tensor[]) -> tf.Tensor[]
        Parameter transformation prior to objective function call.
    learner_mask : bool[]
        Mask indicating which parameters should be trained on, and which
        should use the reference optimizer (policy_ref).
    """

    def __init__(
            self, policy,
            get_objective=None, transform=None, learner_mask=AlwaysTrue()):
        self.policy = policy
        self.get_objective = get_objective
        self.transform = transform
        self.mask = learner_mask

    def create_state(self, params, state=None):
        """Abstracted unroll state for training loop inner optimization.

        Parameters
        ----------
        params : object
            Nested structure of tensors describing initial problems state.
        state : UnrollState or None
            If state is not None, uses the state and global state there.
            Params are replaced.
        """
        # Passthrough
        if state is not None:
            return UnrollState(
                params=params, states=state.states,
                global_state=state.global_state)

        # Proper processing
        states = [
            (self.policy if mask else self.policy_ref).get_initial_state(p)
            for p, mask in zip(params, self.mask)]

        return UnrollState(
            params=params, states=states,
            global_state=self.policy.get_initial_state_global())

    def advance_param(self, args, mask, global_state):
        """Advance a single parameter, depending on mask."""
        if mask:
            return self.policy.call(*args, global_state)
        else:
            return self.policy_ref.call(*args)

    def advance_state(self, unroll_state, batch):
        """Advance this state by a single inner step.

        Parameters
        ----------
        unroll_state : UnrollState
            Unroll state data.
        batch : object
            Nested structure containing data batch.

        Returns
        -------
        float
            Objective value.
        """
        # 1. objective <- get_objective(params, batch)
        with tf.GradientTape() as tape:
            tape.watch(unroll_state.params)
            objective = self.get_objective(
                self.transform(unroll_state.params), batch)
        # 2. grads <- gradient(objective, params)
        grads = tape.gradient(objective, unroll_state.params)
        # 3. delta p, state <- policy(params, grads, local, global)
        dparams, states_new = list(map(list, zip(*[
            self.advance_param(args, mask, unroll_state.global_state)
            for mask, args in zip(
                self.mask,
                zip(unroll_state.params, grads, unroll_state.states))
        ])))
        # 4. p <- p - delta p
        params_new = [p - d for p, d in zip(unroll_state.params, dparams)]
        # 5. global_state <- global_policy(local states, global state)
        global_state_new = self.policy.call_global(
            [s for s, mask in zip(states_new, self.mask) if mask],
            unroll_state.global_state)

        return objective, UnrollState(
            params=params_new, states=states_new,
            global_state=global_state_new)


def state_distance(s1, s2):
    """Compute parameter l2 distance between two states."""
    return tf.add_n([
        tf.nn.l2_loss(self_p - ref_p)
        for self_p, ref_p in zip(s1.params, s2.params)
    ])
