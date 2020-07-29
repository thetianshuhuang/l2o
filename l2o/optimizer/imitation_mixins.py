import tensorflow as tf


class ImitationLossMixin:

    @tf.function
    def imitation_loss(
            self, weights, data, unroll_state,
            unroll=20, problem=None, is_batched=False, teachers=None,
            strategy=tf.math.reduce_mean):
        """Get imitation learning loss.

        The problem must be built in persistent mode for the teacher to use,
        and the caller is responsible for resetting the problem when necessary.

        See ``meta_loss`` for tensorflow quirks / rules.

        Parameters
        ----------
        weights : tf.Tensor
            Array specifying loss weights for each unroll iteration. For
            example, [1 ... 1] indicates total loss, while [1/d ... 1/d]
            indicates mean loss and [0 ... 0 1] final loss.
        data : object
            Nested structure containing data tensors.
        unroll_state : UnrollState
            Starting (params, states, global_state) tuple. If any elements are
            None, fetches from the appropriate ``.get_`` method, but returns
            as None.

        Keyword Args
        ------------
        unroll : int (bound)
            Number of unroll iterations
        problem : problems.Problem (bound)
            Training problem
        is_batched : bool (bound)
            Batch training or full batch training?
        teachers : tf.keras.optimizers.Optimizer[] (bound)
            List of optimizers to train against.
        strategy : Callable (float[] -> float)
            Imitation learning multi-teacher loss strategy. Suggested:
              - ``tf.math.reduce_mean``: classic multi-teacher mean loss.
              - ``tf.math.reduce_max``: minimax loss.

        Returns
        -------
        (tf.Tensor, UnrollState)
            [0] Meta loss
            [1] Final (params, state, global_state) tuple. None values in are
                returned as None values.
        """
        unroll_state, state_mask = self._get_state(problem, unroll_state)

        loss = 0.
        for i in tf.range(unroll):
            batch = [dim[i] for dim in data] if is_batched else data

            # Run learner
            # NOTE: this cannot be split into a method since tensorflow
            # has over-strict loop type checking as of 2.3.0-rc1 that raises
            # an error during autograph conversion whenever global_state is
            # None, even though in this case global_state will be constant.
            with tf.GradientTape() as tape:
                tape.watch(unroll_state.params)
                current_obj = problem.objective(unroll_state.params, batch)
            grads = tape.gradient(current_obj, unroll_state.params)
            unroll_state = self._train_apply_gradients(unroll_state, grads)

            # Run teachers
            for teacher, var_set in zip(teachers, problem.trainable_variables):
                teacher.minimize(
                    lambda: problem.objective(var_set, batch), var_set)

            # Loss for each teacher is l2 between parameters
            # Loss for multi-teacher is determined by the ``strategy``
            d_loss = strategy([
                tf.add_n([
                    tf.nn.l2_loss(svar - tvar)
                    for svar, tvar in zip(unroll_state.params, var_set)
                ]) for var_set in problem.trainable_variables
            ])
            if self.use_log_objective:
                d_loss = tf.math.log(d_loss)
            loss += weights[i] * d_loss

        return loss, self._mask_state(unroll_state, state_mask)

    @tf.function
    def imitation_step(self, opt, *args, **kwargs):
        """Wraps imitation_loss to include gradient calculation inside graph
        mode.

        See ``imitation_loss`` for docstring.

        Parameters
        ----------
        opt : tf.keras.optimizers.Optimizer
            Optimizer to apply step using
        """

        # Specify trainable_variables specifically for efficiency
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.network.trainable_variables)
            loss, unroll_state = self.imitation_loss(*args, **kwargs)

        # Standard apply_gradient paradigam
        # Used instead of ``optimizer.minimize`` to expose the current loss
        grads = tape.gradient(loss, self.network.trainable_variables)
        opt.apply_gradients(zip(grads, self.network.trainable_variables))

        return loss, unroll_state
