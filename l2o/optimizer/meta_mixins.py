import functools

import tensorflow as tf


class MetaLossMixin:

    def _scale_objective(self, objective, initial_obj, weight):
        """Normalizes the objective based on the initial objective value.
        This function is not a @tf.function since it should be wrapped by
        meta_loss, which should handle conversion.
        Parameters
        ----------
        objective : tf.Tensor
            Objective value. Nominally a scalar.
        initial_obj : tf.Tensor
            Initial objective value to normalize by.
        weight : tf.Tensor
            Weight for this objective value.
        Returns
        -------
        tf.Tensor
            Scaled objective according to rules described in initializer
            (use_log_objective, use_numerator_epsilon, epsilon)
        """
        if self.use_log_objective:
            if self.use_numerator_epsilon:
                return weight * (
                    tf.math.log(objective + self.epsilon)
                    - tf.math.log(initial_obj + self.epsilon))
            else:
                return weight * (
                    tf.math.log(objective)
                    - tf.math.log(initial_obj + self.epsilon))
        else:
            return weight * objective / (initial_obj + self.epsilon)

    def _add_noise(self, grads, noise_stddev=0.0):
        """Add normally distributed noise to gradients in order to simulate
        minibatch noise.

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
        """Compute initial objective value

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
        float or tf.TensorArray
            If full batch, returns the initial objective value. If batched,
            returns the initial objective value for each batch.
        """
        # scale_objective -> need to compute initial values
        if self._scale_objective:
            # When batched, use initial values for each batch
            if is_batched:
                init_obj = tf.TensorArray(tf.float32, size=unroll)
                for i in tf.range(unroll):
                    obj = problem.objective(params, [d[i] for d in data])
                    init_obj.write(i, obj)
            else:
                init_obj = problem.objective(params, data)
        # Not scale_objective -> just use 1. as denominator
        else:
            init_obj = 1.

        return init_obj

    def _max_obj(self, init_obj, current_obj):
        """Helper to check for exceeding maximum objective limits"""
        if self.obj_train_max_multiplier > 0:
            max_obj = (
                (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
                + init_obj)
            return max_obj < current_obj
        else:
            return False

    @tf.function
    def meta_loss(
            self, weights, data, unroll_state, unroll=20, problem=None,
            is_batched=False, noise_stddev=0.0):
        """Get meta-training loss.

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
        noise_stddev : float (bound)
            Normal noise to add to gradients.

        Returns
        -------
        (tf.Tensor, UnrollState)
            [0] Meta loss
            [1] Final (params, state, global_state) tuple. None values in are
                returned as None values.
        """
        unroll_state, state_mask = self._get_state(problem, unroll_state)
        init_obj = self._compute_init_obj(
            unroll_state.params, problem, data, unroll, is_batched)

        loss = 0.
        for i in tf.range(unroll):
            batch = [dim[i] for dim in data] if is_batched else data
            init_obj_step = init_obj.read(i) if is_batched else init_obj

            # Compute problem gradient
            with tf.GradientTape() as tape:
                tape.watch(unroll_state.params)
                current_obj = problem.objective(unroll_state.params, batch)
            grads = tape.gradient(current_obj, unroll_state.params)

            # Early termination for exploding objective
            if is_batched and self._max_obj(init_obj_step):
                break

            # Apply gradients
            unroll_state = self._train_apply_gradients(unroll_state, grads)

            # Add to loss
            loss += self._scale_objective(
                current_obj, init_obj_step, weights[i])

            # Check for finite objective
            if not tf.math.is_finite(loss):
                break

        return loss, self._mask_state(unroll_state, state_mask)

    @tf.function
    def meta_step(self, *args, opt=None, **kwargs):
        """Wraps meta_loss to include gradient calculation inside graph mode.

        See ``meta_loss`` for docstring and ``_base_step`` for internal
        mechanism.

        Keyword Args
        ------------
        opt : tf.keras.optimizers.Optimizer
            Optimizer to apply step using
        """
        return self._base_step(opt, self.meta_loss, args, kwargs)
