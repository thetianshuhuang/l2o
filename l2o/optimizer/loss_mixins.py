import tensorflow as tf


class LossMixin:

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

    @tf.function
    def _meta_loss(
            self, problem, weights, unroll,
            params=None, states=None, data=None, noise_stddev=0.0):

        # Fetch parameters, state if not starting from previous state
        if params is None:
            params = problem.get_parameters()
        if states is None:
            states = [self._initialize_state(p) for p in params]

        # Compute initial objective value
        if data is None:
            init_obj = problem.objective(params, None)
        else:
            init_obj = 0
            for i in tf.range(unroll):
                init_obj += problem.objective(params, [dim[i] for dim in data])
            init_obj /= tf.cast(unroll, tf.float32)

        # Optional "reasonable limits" on objective over optimization period
        # If obj_train_max_multiplier is defined > 0, meta-loss calculation
        # will terminate if the loss explodes
        if self.obj_train_max_multiplier > 0:
            max_obj = (
                (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
                + init_obj)

        # Loss accumulator
        loss = 0.

        # cond1: less than ``unroll`` iterations.
        for i in tf.range(unroll):
            # Unbatch
            batch = None if data is None else [dim[i] for dim in data]

            # Compute gradient
            with tf.GradientTape() as tape:
                tape.watch(params)
                current_obj = problem.objective(params, batch)
            grads = tape.gradient(current_obj, params)
            # cond3: objective is a reasonable multiplier of the original
            if self.obj_train_max_multiplier > 0 and current_obj > max_obj:
                break
            # Optionally add artificial noise
            grads = self._add_noise(grads, noise_stddev=noise_stddev)

            # Apply gradients
            params, states = list(zip(*[
                self._compute_update(z) for z in zip(params, grads, states)]))

            # Add to loss
            loss += self._scale_objective(current_obj, init_obj, weights[i])
            # cond2: objective is still finite
            if not tf.math.is_finite(loss):
                break

        return loss, params, states

    @tf.function
    def meta_loss(
            self, problem, weights, unroll,
            params=None, states=None, data=None, noise_stddev=0.0):

        # Fetch parameters, state if not starting from previous state
        if params is None:
            params = problem.get_parameters()
        if states is None:
            states = [self._initialize_state(p) for p in params]

        # Compute initial objective value
        # if data is None:
        #     init_obj = problem.objective(params, None)
        # else:
        #     init_obj = 0
        #     for i in tf.range(unroll):
        #         init_obj += problem.objective(params, [dim[i] for dim in data])
        #     init_obj /= tf.cast(unroll, tf.float32)

        # # Optional "reasonable limits" on objective over optimization period
        # # If obj_train_max_multiplier is defined > 0, meta-loss calculation
        # # will terminate if the loss explodes
        # if self.obj_train_max_multiplier > 0:
        #     max_obj = (
        #         (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
        #         + init_obj)

        # Loss accumulator
        loss = 0.

        # cond1: less than ``unroll`` iterations.
        for i in tf.range(unroll):
            # Unbatch
            batch = None if data is None else [dim[i] for dim in data]

            # Compute gradient
            with tf.GradientTape() as tape:
                tape.watch(params)
                current_obj = problem.objective(params, batch)
            grads = tape.gradient(current_obj, params)
            # cond3: objective is a reasonable multiplier of the original
            # if self.obj_train_max_multiplier > 0 and current_obj > max_obj:
            #     break
            # Optionally add artificial noise
            # grads = self._add_noise(grads, noise_stddev=noise_stddev)

            # Apply gradients
            params, states = list(zip(*[
                self._compute_update(z) for z in zip(params, grads, states)]))

            # Add to loss
            loss += current_obj
            # cond2: objective is still finite
            # if not tf.math.is_finite(loss):
            #     break

        return loss, params, states
