import tensorflow as tf


class LossMixin:

    def _get_state(self, problem, params=None, states=None, global_state=None):
        """Helper function to initialize or use existing states.

        Each argument is generated as appropriate if None.

        Parameters
        ----------
        problem : problems.Problem
            Current problem

        Keyword Args
        ------------
        params : tf.Tensor[]
            List of problem parameters.
        states : object
            Nested structure containing state tensors.
        global_state : object
            Nested structure containing global state information.

        Returns
        -------
        (tf.Tensor[], object, object)
            [0] params or generated params
            [1] states or generated states
            [2] global_state or generated global_state or None (if no
                ``get_initial_state_global`` method)
        """
        if params is None:
            params = problem.get_parameters()
        if states is None:
            states = [self._initialize_state(p) for p in params]
        if global_state is None:
            if hasattr(self.network, "get_initial_state_global"):
                global_state = self.network.get_initial_state_global()

        return params, states, global_state

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

    def _step(
            self, learning_state, problem, batch, noise_stddev):
        """Helper function to run a single optimization step"""

        # Unpack
        params, states, global_state = learning_state

        # Compute gradient
        with tf.GradientTape() as tape:
            tape.watch(params)
            current_obj = problem.objective(params, batch)
        grads = tape.gradient(current_obj, params)

        # Optionally add artificial noise
        grads = self._add_noise(grads, noise_stddev=noise_stddev)

        # Apply gradients
        params, states = list(map(list, zip(*[
            self._compute_update(*z) for z in zip(params, grads, states)
        ])))
        if global_state is not None:
            self.network.call_global(states, global_state)

        return current_obj, (params, states, global_state)

    @tf.function
    def meta_loss(
            self, weights, data, params=None, states=None, global_state=None,
            unroll=20, problem=None, is_batched=False, noise_stddev=0.0):
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
            Tensor specifying loss weights for each unroll iteration. For
            example, [1 ... 1] indicates total loss, while [1/d ... 1/d]
            indicates mean loss and [0 ... 0 1] final loss.
        data : object
            Nested structure containing data tensors.

        Keyword Args
        ------------
        params : tf.Tensor[] (optional)
            List of problem parameters. If None, is generated each time.
        states : object (optional)
            Nested structure containing state tensors.
        global_state : object (optional)
            Nested structure containing global state information.
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
        (tf.Tensor, tf.Tensor[], object)
            [0] Meta loss
            [1] Final parameters
            [2] Final state
        """
        # (params, states, global_state) triple
        learning_state = self._get_state(
            problem, params=params, states=states, global_state=global_state)

        # Compute initial objective value
        if is_batched:
            init_obj = 0.
            for i in tf.range(unroll):
                init_obj += problem.objective(params, [dim[i] for dim in data])
            init_obj /= tf.cast(unroll, tf.float32)
        else:
            init_obj = problem.objective(params, data)

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
            batch = [dim[i] for dim in data] if is_batched else data

            # The actual step
            current_obj, learning_state = self._step(
                learning_state, problem, batch, noise_stddev)
            # cond3: objective is a reasonable multiplier of the original
            if self.obj_train_max_multiplier > 0 and current_obj > max_obj:
                break

            # Add to loss
            loss += self._scale_objective(current_obj, init_obj, weights[i])
            # cond2: objective is still finite
            if not tf.math.is_finite(loss):
                break

        return (loss, *learning_state)

    @tf.function
    def imitation_loss(
            self, weights, data, params=None, states=None, global_state=None,
            unroll=20, problem=None, is_batched=False, teachers=None):
        """Get imitation learning loss.

        The problem must be built in persistent mode for the teacher to use,
        and the caller is responsible for resetting the problem when necessary.

        See ``meta_loss`` for tensorflow quirks / rules.

        Parameters
        ----------
        weights : tf.Tensor
            Tensor specifying loss weights for each unroll iteration. For
            example, [1 ... 1] indicates total loss, while [1/d ... 1/d]
            indicates mean loss and [0 ... 0 1] final loss.
        data : object
            Nested structure containing data tensors.

        Keyword Args
        ------------
        params : tf.Tensor[] (optional)
            List of problem parameters. If None, is generated each time.
        states : object (optional)
            Nested structure containing state tensors.
        global_state : object (optional)
            Nested structure containing global state information.
        unroll : int (bound)
            Number of unroll iterations
        problem : problems.Problem (bound)
            Training problem
        is_batched : bool (bound)
            Batch training or full batch training?
        teachers : tf.keras.optimizers.Optimizer[] (bound)
            List of optimizers to train against.

        Returns
        -------
        (tf.Tensor, tf.Tensor[], object)
            [0] Meta loss
            [1] Final parameters
            [2] Final state
        """
        # (params, states, global_state) triple
        learning_state = self._get_state(
            problem, params=params, states=states, global_state=global_state)

        # Loss accumulator
        loss = 0.

        # cond1: less than ``unroll`` iterations.
        for i in tf.range(unroll):
            # Unbatch
            batch = [dim[i] for dim in data] if is_batched else data

            # Run learner
            _, learning_state = self._step(learning_state, problem, batch, 0.0)

            # Run teacher
            _vars = problem.trainable_variables
            teachers[0].minimize(
                lambda: problem.objective(_vars, batch), _vars)

            # Loss is l2 between parameters
            loss += weights[i] * tf.add_n([
                tf.nn.l2_loss(student - teacher)
                for student, teacher in zip(learning_state[0], _vars)
            ])

        return (loss, *learning_state)
