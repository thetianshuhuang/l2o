"""Inner training and outer loss computation."""

import collections
import tensorflow as tf

from .random_scaling import create_random_parameter_scaling


UnrollState = collections.namedtuple(
    "UnrollState", ['params', 'states', 'global_state'])


class LossMixin:
    """Inner Training and Outer Loss computation Mixin."""

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
            return (
                tf.math.log(objective + self.epsilon)
                - tf.math.log(initial_obj + self.epsilon))
        else:
            return objective / (initial_obj + self.epsilon)

    def _compute_init_obj(self, params, problem, data, unroll):
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

        Returns
        -------
        float or tf.Tensor
            If full batch, returns the initial objective value. If batched,
            returns the initial objective value for each batch.
        """
        # scale_objective -> need to compute initial values
        if self.scale_objective:
            # When batched, use initial values for each batch
            ctx = tf.distribute.get_replica_context()
            return tf.stack([
                ctx.all_reduce(
                    tf.distribute.ReduceOp.SUM,
                    problem.objective(params, [dim[i] for dim in data])
                ) / problem.batch_size
                for i in range(unroll)])
        # Not scale_objective -> just use 1. as denominator
        else:
            return tf.tile([1.], [unroll])

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

    def _train_apply_gradients(self, unroll_state, grads):
        """Helper function to apply gradients.

        1. delta p, state <- network(
            params, gradients, local state, global state)
        2. p <- p - delta p
        3. global_state <- global_network(local states, global state)
        """
        dparams, states = list(map(list, zip(*[
            self.network.call(*z, unroll_state.global_state)
            for z in zip(unroll_state.params, grads, unroll_state.states)
        ])))
        params = [p - d for p, d in zip(unroll_state.params, dparams)]

        global_state = self.network.call_global(
            states, unroll_state.global_state)

        return UnrollState(params, states, global_state)

    def _get_objective_grads(self, get_objective, params, batch, batch_size):
        """Helper to compute objective and gradients with distribute."""
        ctx = tf.distribute.get_replica_context()

        with tf.GradientTape() as tape:
            tape.watch(params)
            objective = get_objective(params, batch)
        grads = tape.gradient(objective, params)

        objective = ctx.all_reduce(tf.distribute.ReduceOp.SUM, objective) / batch_size
        grads = ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads) / batch_size

        return objective, grads

    def _learner_imitation_loss(
            self, get_objective, params, batch, problem_vars):
        """Helper to compute inner imitation loss."""
        # Run teachers
        for teacher, var_set in zip(self.teachers, problem_vars):
            teacher_obj, grads = self._get_objective_grads(
                get_objective, var_set, batch, batch_size)
            teacher.apply_gradients(zip(grads, var_set))

        # Loss for each teacher is l2 between parameters
        # Loss for multi-teacher is determined by ``self.loss_reduce``
        teacher_loss = [
            tf.add_n([
                tf.nn.l2_loss(svar - tvar)
                for svar, tvar in zip(params, var_set)
            ]) for var_set in problem_vars
        ]
        if self.use_log_objective:
            return teacher_loss, tf.math.log(self.loss_reduce(teacher_loss))
        else:
            return teacher_loss, self.loss_reduce(teacher_loss)

    # @tf.function
    def abstract_loss(
            self, data, params, unroll=20, problem=None,
            meta_loss_weight=0.0, imitation_loss_weight=1.0, seed=None):
        """Get abstract imitation learning and meta learning loss.

        Runs inner training in order to compute the abstract loss
        ```
        strategy(imitation_loss(teachers)) + meta_loss_weight * meta_loss().
        ```

        NOTE: The problem must be built in persistent mode for the teacher to
        use. This method will modify the persistent-mode teacher variables
        owned by the problem.

        By decorating as a @tf.function, the for loop is wrapped into a
        tf.while_loop. See `https://www.tensorflow.org/guide/function`.

        The following rules must be followed:
         -  No modifying unroll_state. The new state must be a new copy of the
            object on each loop.
         -  No variable creation. Variable creation is only allowed in eager
            mode.
         -  No ``tf.Variable``s may be assigned, since this stops gradients.
            This precludes the use of tf.keras.Model in training problems, as
            well as usage of the ``_create_slots`` system.

        Parameters
        ----------
        data : object
            Nested structure containing data tensors.
        params : tf.Tensor[]
            Initial problem parameter values.

        Keyword Args
        ------------
        unroll : int
            Number of unroll iterations
        problem : problems.Problem
            Training problem
        meta_loss_weight : float
            Weight applied to meta loss. If 0, meta loss is not computed.
        imitation_loss_weight : float
            Weight applied to imitation loss. If 0, imitation loss is not
            computed.
        seed : int or None
            Seed to use for intializing parameters.

        Returns
        -------
        (tf.Tensor, tf.Tensor[], tf.Tensor{})
            [0] Meta loss.
            [1] Final problem parameters.
            [2] Summary statistics collected by self.step_callback if present.
        """

        # Split data; data dimensions are ``[unroll, batch] + [data]``
        data = [
            tf.stack(tf.split(dim, num_or_size_splits=unroll))
            for dim in data]

        # Make random scaling
        params, transform = create_random_parameter_scaling(
            params, spread=self.parameter_scale_spread, seed=seed)
        # Calculate initial objective values for scaling
        init_obj = self._compute_init_obj(params, problem, data, unroll)
        # Prepare problem tf.Variables
        problem.reset(values=params)
        # Make UnrollState
        unroll_state = UnrollState(
            params, [self.network.get_initial_state(p) for p in params],
            self.network.get_initial_state_global())

        def get_objective(params, batch_):
            return problem.objective(transform(params), batch_)

        # Loop placeholder
        teacher_losses = [tf.zeros(shape=()) for _ in self.teachers]

        loss = 0.
        # step_log = self.step_callback(unroll, len(self.teachers))
        for i in tf.range(unroll):
            weight = self.unroll_weight(i, unroll)
            batch = [dim[i] for dim in data]

            # Run learner
            current_obj, grads = self._get_objective_grads(
                get_objective, unroll_state.params, batch, problem.batch_size)
            unroll_state = self._train_apply_gradients(unroll_state, grads)
            # Check for exploding loss
            if self._max_obj(init_obj[i], current_obj):
                break

            # Optionally add imitation loss
            if imitation_loss_weight > 0.0:
                teacher_losses, il_loss = self._learner_imitation_loss(
                    get_objective, unroll_state.params, batch,
                    problem.trainable_variables)
                loss += weight * imitation_loss_weight * il_loss
            # Optionally add meta loss
            if meta_loss_weight > 0.0:
                loss += (
                    weight * meta_loss_weight
                    * self._scale_meta_objective(current_obj, init_obj[i]))

            # Log optional statistics
            # step_log.on_step_end(i, current_obj, teacher_losses)

        return loss, params  # , step_log.summarize()

    @tf.function
    def _abstract_loss(self, *args, **kwargs):
        return self.abstract_loss(*args, **kwargs)
