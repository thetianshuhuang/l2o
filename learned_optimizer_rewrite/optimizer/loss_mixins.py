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
                    tf.log(objective + self.epsilon)
                    - tf.log(initial_obj + self.epsilon))
            else:
                return weight * (
                    tf.log(objective) - tf.log(initial_obj + self.epsilon))
        else:
            return weight * objective / (initial_obj + self.epsilon)

    @tf.function
    def meta_loss(self, problem, weights, data=None):
        """Meta training loss

        The caller is responsible for setting the initial values of the problem
        parameters, which are owned by `problem`.

        By decorating as @tf.function, the for loop should be wrapped into
        a tf.while_loop. See `https://www.tensorflow.org/guide/function`.

        Parameters
        ----------
        problem : problems.Problem
            Optimizee module. Should have a trainable_variables @property and a
            .objective() method, and should own its own parameters.
        weights : tf.Tensor
            Tensor specifying loss weights. The dimensionality specifies the
            number of unrolls. For example, [1 ... 1] indicates total loss,
            while [1/d ... 1/d] indicates mean loss and [0 ... 0 1] final loss.

        Keyword Args
        ------------
        data : tf.Tensor[] | None
            Input data, with size of batch_size * unroll.

        Returns
        -------
        tf.Tensor
            Scalar meta loss value
        """

        loss = 0.
        unroll = tf.size(weights)

        # Compute init_obj as mean over minibatches if dataset is available
        if data is None:
            init_obj = problem.objective()
        else:
            batches = list(zip(
                *[tf.split(dim, num_or_size_splits=unroll) for dim in data]))
            init_obj = tf.reduce_mean(
                [problem.objective(batch) for batch in batches])

        # Optional "reasonable limits" on objective over optimization period
        # If obj_train_max_multiplier is defined > 0, meta-loss calculation
        # will terminate if the loss explodes
        if self.obj_train_max_multiplier > 0:
            max_obj = (
                (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
                + init_obj)

        # Create new slots
        # Should reset values if they already exist
        # Should also initialize state
        self._create_slots(problem.trainable_variables)

        for i in range(unroll):
            weight = weights[i]

            # cond2: objective is still finite
            if not tf.math.is_finite(loss):
                break

            batch = None if data is None else batches[i]
            current_obj = problem.objective(batch)

            # cond3: objective is a reasonable multiplier of the original
            if self.obj_train_max_multiplier > 0 and current_obj > max_obj:
                break

            # call this optimizer on the problem
            # outside in the meta-training loop, we will call
            # minimize(meta_loss, self.trainable_variables)

            # this calls self._compute_update via self._apply_dense
            self.minimize(
                lambda: problem.objective(batch),
                problem.trainable_variables)

            # Add to loss
            loss += self._scale_objective(current_obj, init_obj, weight)

        # @tf.function should compile this down as per tensorflow 2 best
        # practices
        return loss

    @tf.function
    def imitation_loss(
            self, student_cpy, teacher_cpy, teacher, weights, data=None):
        """Imitation learning loss

        Parameters
        ----------
        student_cpy : problems.Problem
            Optimizee module; student copy. Contains variables that student
            optimizes on.
        teacher_cpy : problems.Problem
            Optimizee module; teacher copy. Contains variables that teacher
            optimizes on. Should be an exact copy of student_cpy
            (via Problem.clone_problem())
        teacher : tf.keras.optimizers.Optimizer
            Teacher optimizer to imitate
        weights : tf.Tensor
            Tensor specifying loss weights. The dimensionality specifies the
            number of unrolls.

        Keyword Args
        ------------
        data : tf.Tensor[] | None
            Input data, with size of batch_size * unroll.

        Returns
        -------
        tf.Tensor
            Scalar imitation loss value
        """

        loss = 0.
        unroll = tf.size(weights)

        # Split batches between unrolls if needed
        if data is not None:
            batches = list(zip(
                *[tf.split(dim, num_or_size_splits=unroll) for dim in data]))

        for i in range(unroll):
            weight = weights[i]

            # Run single step on student and teacher
            self.minimize(
                lambda: student_cpy.objective(batches[i]),
                student_cpy.trainable_variables)
            teacher.minimize(
                lambda: teacher_cpy.objective(batches[i]),
                teacher_cpy.trainable_variables)

            # Loss is l2 between parameter state
            losses = [
                tf.nn.l2_loss(student - teacher)
                for student, teacher in zip(
                    student_cpy.trainable_variables,
                    teacher_cpy.trainable_variables)
            ]
            loss += weight * tf.add_n(losses)

        return loss
