import collections

UnrollState = collections.namedtuple(
    "UnrollState", ['params', 'states', 'global_state'])


class StateMixin:

    def _train_apply_gradients(self, unroll_state, grads):
        """Helper function to apply gradients.

        Parameters
        ----------
        unroll_state : UnrollState
            (params, states, global_state) tuple
        grads : tf.Tensor[]
            List of gradients corresponding to params.

        Returns
        -------
        UnrollState
            New loop state
        """
        params, states = list(map(list, zip(*[
            self._compute_update(*z) for z in zip(
                unroll_state.params, grads, unroll_state.states)
        ])))
        if unroll_state.global_state is not None:
            global_state = self.network.call_global(
                states, unroll_state.global_state)

        return UnrollState(
            params=params, states=states, global_state=global_state)

    def _get_state(self, problem, unroll_state):
        """Helper function to initialize or use existing states

        Parameters
        ----------
        problem : problems.Problem
            Current problem to .get_parameters() for
        unroll_state : UnrollState
            Unroll state to initialize from. If any elements are None, fetches
            from the appropriate ``.get_`` method.

        Returns
        -------
        (UnrollState, UnrollState)
            [0] State with None values initialized
            [1] Mask, with elements set to True if initially not None and False
                if None.
        """
        mask = UnrollState(*[s is not None for s in unroll_state])

        params, states, global_state = unroll_state
        if params is None:
            params = problem.get_parameters()
        if states is None:
            states = [self._initialize_state(p) for p in params]
        if global_state is None:
            global_state = self.network.get_initial_state_global()

        return UnrollState(params, states, global_state), mask

    def _make_unroll_state(
            self, problem, params=None, states=None, global_state=None):
        """Helper function to selectively generate UnrollState tuple"""
        if params:
            params = problem.get_parameters()
        if states:
            states = [self._initialize_state(p) for p in params]
        if global_state:
            global_state = self.network.get_initial_state_global()

        return UnrollState(params, states, global_state)

    def _mask_state(self, unroll_state, mask):
        """Helper function to mask state to return None based on saved mask"""
        return UnrollState(*[
            s if m else None for s, m in zip(unroll_state, mask)])
