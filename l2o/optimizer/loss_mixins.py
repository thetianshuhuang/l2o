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
    def meta_loss(self, problem, weights, unroll, data=None, noise_stddev=0.0):
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
        unroll : tf.constant
            Passsed separately so it can be cast as a tf.constant.

        Keyword Args
        ------------
        data : tf.Tensor[][unroll] | None
            Input data; pre-split into a list of ``unroll`` tuples, with each
            tuple representing a batch.
        noise_stddev : tf.Tensor | float
            Normally distributed noise to add to optimizee gradients; use to
            simulate minibatch noise for full-batch problems.

        Returns
        -------
        tf.Tensor
            Scalar meta loss value
        """

        # Compute init_obj as mean over minibatches if dataset is available
        if data is None:
            init_obj = problem.objective(None)
        else:
            init_obj = 0.
            for i in tf.range(unroll):
                init_obj += problem.objective([dim[i] for dim in data])
            init_obj /= tf.cast(unroll, tf.float32)

        # Optional "reasonable limits" on objective over optimization period
        # If obj_train_max_multiplier is defined > 0, meta-loss calculation
        # will terminate if the loss explodes
        if self.obj_train_max_multiplier > 0:
            max_obj = (
                (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
                + init_obj)

        loss = 0.

        # cond1: less than unroll iterations.
        for i in tf.range(unroll):
            batch = None if data is None else [dim[i] for dim in data]

            # cond2: objective is still finite
            if not tf.math.is_finite(loss):
                break

            # Compute gradient
            with tf.GradientTape() as tape:
                current_obj = problem.objective(batch)
            grad = tape.gradient(current_obj, problem.trainable_variables)

            # Optionally add artificial noise
            grad = self._add_noise(grad, noise_stddev=noise_stddev)

            # cond3: objective is a reasonable multiplier of the original
            if self.obj_train_max_multiplier > 0 and current_obj > max_obj:
                break

            # Apply gradients
            # this calls self._compute_update via self._apply_dense
            self.apply_gradients(zip(grad, problem.trainable_variables))

            # Add to loss
            loss += self._scale_objective(current_obj, init_obj, weights[i])

        # @tf.function should compile this down as per tensorflow 2 best
        # practices
        return loss

    @tf.function
    def imitation_loss(
            self, student_cpy, teacher_cpy, teacher, weights, unroll,
            data=None, noise_stddev=0.0):
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
        unroll : tf.constant
            Passsed separately so it can be cast as a tf.constant.

        Keyword Args
        ------------
        data : tf.Tensor[] | None
            Input data, with size of batch_size * unroll.
        noise_stddev : tf.Tensor | float
            Normally distributed noise to add to optimizee gradients; use to
            simulate minibatch noise for full-batch problems.

        Returns
        -------
        tf.Tensor
            Scalar imitation loss value
        """

        loss = 0.

        for i in tf.range(unroll):
            batch = None if data is None else [dim[i] for dim in data]

            # Compute gradient on same batch
            with tf.GradientTape() as tape:
                student_obj = student_cpy.objective(batch)
                teacher_obj = teacher_cpy.objective(batch)
            student_grad = tape.gradient(
                student_obj, student_cpy.trainable_variables)
            teacher_grad = tape.gradient(
                teacher_obj, teacher_cpy.trainable_variables)

            # Optionally add artificial noise
            student_grad = self._add_noise(
                student_grad, noise_stddev=noise_stddev)
            teacher_grad = self._add_noise(
                teacher_grad, noise_stddev=noise_stddev)

            # Run single step on student and teacher
            self.apply_gradients(
                student_grad, student_cpy.trainable_variables)
            teacher.apply_gradients(
                teacher_grad, teacher_cpy.trainable_variables)

            # Loss is l2 between parameter state
            losses = [
                tf.nn.l2_loss(student - teacher)
                for student, teacher in zip(
                    student_cpy.trainable_variables,
                    teacher_cpy.trainable_variables)
            ]
            loss += weights[i] * tf.add_n(losses)

        return loss
