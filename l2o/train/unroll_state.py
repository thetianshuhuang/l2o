"""Abstracted unroll state for training loop inner optimization."""

import tensorflow as tf
import collections

UnrollState = collections.namedtuple(
    "UnrollState", ["params", "states", "global_state"])


def create_state(params, policy):
    """Abstracted unroll state for training loop inner optimization.

    Since tensorflow's behavior related to objects in tf.function is quite
    broken, UnrollState is implemented just like a class -- just not as one.

    Furthermore, class attributes can only contain tensors, not bound methods
    since tensorflow's loop logic is also super scuffed and tries to interpret
    all variables that *might* be states as tensors.

    Parameters
    ----------
    params : object
        Nested structure of tensors describing initial problems state.
    policy : l2o.policies.BaseLearnToOptimizePolicy
        Optimization gradient policy to apply.
    """
    return UnrollState(
        params=params,
        states=[policy.get_initial_state(p) for p in params],
        global_state=policy.get_initial_state_global())


def advance_state(unroll_state, batch, get_objective, transform, policy):
    """Advance this state by a single inner step.

    Parameters
    ----------
    unroll_state : UnrollState
        Unroll state data.
    batch : object
        Nested structure containing data batch.
    get_objective : callable(params, batch) -> float
        Computes objective value from current parameters and data batch.
    transform : callable(tf.Tensor[]) -> tf.Tensor[]
        Parameter transformation prior to objective function call.
    policy : l2o.policies.BaseLearnToOptimizePolicy
        Optimization gradient policy to apply.

    Returns
    -------
    float
        Objective value.
    """
    # 1. objective <- get_objective(params, batch)
    with tf.GradientTape() as tape:
        tape.watch(unroll_state.params)
        objective = get_objective(transform(unroll_state.params), batch)
    # 2. grads <- gradient(objective, params)
    grads = tape.gradient(objective, unroll_state.params)
    # 3. delta p, state <- policy(params, grads, local state, global state)
    dparams, states_new = list(map(list, zip(*[
        policy.call(*z, unroll_state.global_state)
        for z in zip(unroll_state.params, grads, unroll_state.states)
    ])))
    # 4. p <- p - delta p
    params_new = [p - d for p, d in zip(unroll_state.params, dparams)]
    # 5. global_state <- global_policy(local states, global state)
    global_state_new = policy.call_global(
        states_new, unroll_state.global_state)

    return objective, UnrollState(
        params=params_new, states=states_new, global_state=global_state_new)


def state_distance(s1, s2):
    """Compute parameter l2 distance between two states."""
    return tf.add_n([
        tf.nn.l2_loss(self_p - ref_p)
        for self_p, ref_p in zip(s1.params, s2.params)
    ])
