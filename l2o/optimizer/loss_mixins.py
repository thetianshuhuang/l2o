"""Loss computation."""
import tensorflow as tf


class LossMixin:
    """Loss Computation Mixin."""

    def _scale_meta_objective(self, objective, initial_obj):
        """Normalizes the objective based on the initial objective value.

        This function is not a @tf.function since it should be wrapped by
        meta_loss, which should handle conversion.

        Parameters
        ----------
        objective : tf.Tensor
            Objective value. Nominally a scalar.
        initial_obj : tf.Tensor
            Initial objective value to normalize by.

        Returns
        -------
        tf.Tensor
            Scaled objective according to rules described in initializer
            (use_log_objective, use_numerator_epsilon, epsilon)
        """
        if self.use_log_objective:
            if self.use_numerator_epsilon:
                num = tf.math.log(objective + self.epsilon)
            else:
                num = tf.math.log(objective)
            return num - tf.math.log(initial_obj + self.epsilon)
        else:
            return objective / (initial_obj + self.epsilon)

    def _add_noise(self, grads, noise_stddev=0.0):
        """Add normally distributed noise to gradients.

        This simulates minibatch noise in otherwise full-batch problems.

        Parameters
        ----------
        grads : tf.Tensor
            Gradients to add noise to
        noise_stddev : tf.Tensor | float
            Noise stddev; if 0, no noise is added.
        """
        if noise_stddev > 0:
            return [
                g + tf.random.normal(g.shape(), stddev=noise_stddev)
                for g in grads
            ]
        else:
            return grads

    def _compute_init_obj(self, params, problem, data, unroll, is_batched):
        """Compute initial objective value.

        Parameters
        ----------
        params : tf.Tensor[]
            Initial problem parameters
        problem : problems.Problem
            Training problem
        data : tf.TensorArray[] or tf.Tensor[]
            List of TensorArrays containing training batches or list of tensors
            containing full batch data
        unroll : tf.Tensor()
            0-dimension tensor containing unroll size
        is_batched : bool
            Indicates batched or full-batch data

        Returns
        -------
        float or tf.Tensor
            If full batch, returns the initial objective value. If batched,
            returns the initial objective value for each batch.
        """
        # scale_objective -> need to compute initial values
        if self.scale_objective:
            # When batched, use initial values for each batch
            if is_batched:
                return tf.stack([
                    problem.objective(params, [d[i] for d in data])
                    for i in range(unroll)])
            else:
                return tf.tile([problem.objective(params, data)], unroll)
        # Not scale_objective -> just use 1. as denominator
        else:
            return tf.tile([1.], unroll)

    def _max_obj(self, init_obj, current_obj):
        """Helper to check for exceeding maximum objective limits."""
        # obj_train_max_multiplier * init_obj
        if self.obj_train_max_multiplier > 0:
            max_obj = (
                (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
                + init_obj)
            return max_obj < current_obj
        # Plain old infinity
        else:
            return not tf.math.is_finite(current_obj)

    @tf.function
    def abstract_loss(
            self, weights, data, unroll_state,
            unroll=20, problem=None, is_batched=False,
            teachers=[], meta_loss_weight=0.0, imitation_loss_weight=1.0,
            parameter_scale_spread=0.0,
            strategy=tf.math.reduce_mean,
            seed=None):
        """Get abstract imitation learning and meta learning loss.

        Computes the abstract loss
        ```
        strategy(imitation_loss(teachers)) + meta_loss_weight * meta_loss()
        ```

        The problem must be built in persistent mode for the teacher to use,
        and the caller is responsible for resetting the problem when necessary.

        By decorating as a @tf.function, the for loop is wrapped into a
        tf.while_loop. See `https://www.tensorflow.org/guide/function`.

        The following rules must be followed:
         -  No modifying ``states``. The new state must be a copy of the object
            on each loop.
         -  No variable creation. Variable creation is only allowed in eager
            mode.
         -  No ``tf.Variable``s may be assigned, since this stops gradients.
            This precludes the use of tf.keras.Model in training problems, as
            well as usage of the ``_create_slots`` system.

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
        meta_loss_weight : float
            Weight applied to meta loss. If 0, meta loss is not computed.
        imitation_loss_weight : float
            Weight applied to imitation loss. If 0, imitation loss is not
            computed.
        strategy : Callable (float[] -> float)
            Imitation learning multi-teacher loss strategy. Suggested:
              - ``tf.math.reduce_mean``: classic multi-teacher mean loss.
              - ``tf.math.reduce_max``: minimax loss.
        parameter_scale_spread : float
            Each parameter is randomly scaled by a factor sampled from a
            log uniform distribution exp(Unif([-L, L])). If the spread is 0,
            this is equivalent to a constant scale of 1.
        seed : int or None
            Seed to use for intializing parameters.

        Returns
        -------
        (tf.Tensor, UnrollState)
            [0] Meta loss
            [1] Final (params, state, global_state) tuple. None values in are
                returned as None values.
        """
        unroll_state, state_mask = self._get_state(
            problem, unroll_state, seed=seed)
        init_obj = self._compute_init_obj(
            unroll_state.params, problem, data, unroll, is_batched)
        unroll_state, scale = self._make_random_scale(unroll_state, spread)

        loss = 0.
        for i in tf.range(unroll):
            batch = [dim[i] for dim in data] if is_batched else data

            def get_objective(params):
                return problem.objective(
                    [p * s for p, s in zip(params, scale)], batch)

            # Run learner
            with tf.GradientTape() as tape:
                tape.watch(unroll_state.params)
                current_obj = get_objective(unroll_state.params)
            grads = gradient_scale * tape.gradient(
                current_obj, unroll_state.params)
            unroll_state = self._train_apply_gradients(unroll_state, grads)
            # Check for exploding loss
            if self._max_obj(init_obj[i], current_obj):
                break

            # Optionally add imitation loss
            if imitation_loss_weight > 0.0:
                # Run teachers
                trainables = problem.trainable_variables
                for teacher, var_set in zip(teachers, trainables):
                    teacher.minimize(
                        lambda: get_objective(var_set), var_set)
                # Loss for each teacher is l2 between parameters
                # Loss for multi-teacher is determined by the ``strategy``
                il_loss = strategy([
                    tf.add_n([
                        tf.nn.l2_loss(svar - tvar)
                        for svar, tvar in zip(unroll_state.params, var_set)
                    ]) for var_set in trainables
                ])
                if self.use_log_objective:
                    il_loss = tf.math.log(il_loss)
                loss += weights[i] * imitation_loss_weight * il_loss
            # Optionally add meta loss
            if meta_loss_weight > 0.0:
                loss += (
                    weights[i] * meta_loss_weight
                    * self._scale_meta_objective(current_obj, init_obj[i]))

        return loss, self._mask_state(unroll_state, state_mask)
