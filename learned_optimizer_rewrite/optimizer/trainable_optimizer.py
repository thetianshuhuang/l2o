import tensorflow as tf


class TrainableOptimizer(tf.keras.optimizers.Optimizer):

    def __init__(
            self, name, state_keys,
            use_attention=False, use_log_objective=False,
            obj_train_max_multiplier=-1, use_second_derivatives=True,
            use_numerator_epsilon=False, epsilon=1e-6, **kwargs):
        """Initializes the optimizer with the given name and settings.

        Parameters
        ----------
        name : str
            Optimizer name
        state_keys : str[]
            Names of any required state variables. Note that state variables
            are unique across optimization problems, so trainable weights
            should not go here.

        Keyword Args
        ------------
        use_attention : bool
            Whether this optimizer uses attention
        use_log_objective : bool
            Whether this optimizer uses the logarithm of the objective when
            computing the loss
        obj_train_max_multiplier : float
            The maximum multiplier for the increase in the objective before
            meta-training is stopped. If <= 0, meta-training is not stopped
            early.
        use_second_derivatives : bool
            Whether this optimizer uses second derivatives in meta-training.
            This should be set to False if some second derivatives in the
            meta-training problem set are not defined in tensorflow.
        use_numerator_epsilon : bool
            Whether to use epsilon in the numerator when scaling the
            problem objective during meta-training.
        epsilon : float
            Epsilon value.
        """

        self.use_second_derivatives = use_second_derivatives
        self.state_keys = sorted(state_keys)
        self.use_attention = use_attention
        self.use_log_objective = use_log_objective
        self.obj_train_max_multiplier = obj_train_max_multiplier
        self.use_numerator_epsilon = use_numerator_epsilon
        self.epsilon = epsilon

        super(TrainableOptimizer, self).__init__(name)

    def _create_slots(self, var_list):
        """Create slots function required by tf.keras.optimizers.Optimizer

        Parameters
        ----------
        var_list : tf.Variable[]
            List of variables to be optimized on; passed by parent.
        """
        for var in var_list:
            init_states = self._initialize_state(var)
            for name, initial in init_states:
                # add_slot does nothing if the given var/name are already
                # present. This forces reinitialization.
                self.add_slot(var, name).assign(initial)

    def _initialize_state(self, var):
        """Initialize any states required for this variable.

        Parameters
        ----------
        var : tf.Variable
            Tensor containing parmeters to be optimized.

        Returns
        -------
        dict
            Keys: str name of each slot;
            Values: tf.Tensor with initial value
        """
        return {}

    def _initialize_global_state(self):
        """todo"""
        return []

    def _resource_apply_dense(self, grad, var, apply_state):
        """Apply optimizer updates to variables.

        Note: this should only get called via _apply_dense or _apply_sparse
        when using the optimizer via optimizer.minimize or
        optimizer.apply_gradients. During meta-training, the optimizer.train
        function should be used to construct an optimization path that is
        differentiable.

        Parameters
        ----------
        grad : tf.Tensor
            Gradient tensor
        var : tf.Variable
            Variable containing weights that grad is computed for. Should have
            the same shape as grad.
        apply_state : dict
            Ignored for now.

        Returns
        -------
        tf.Operation
            Tensorflow operation that assigns new values to the variable and
            defines dependencies (used for control flow)
        """
        state = {key: self.get_slot(var, key) for key in self.get_slot_names()}
        new_var, new_state = self._compute_update(var, grad, state)

        state_assign_ops = [
            tf.assign(state_var, new_state[key])
            for key, state_var in state.items()
        ]
        with tf.control_dependencies(state_assign_ops):
            update_op = var.assign_add(new_var)

        return update_op

    def _resource_update_sparse(self, grad, var):
        raise NotImplementedError()

    def _compute_update(self, param, grad, state):
        """Computes the update step for optimization.

        Parameters
        ----------
        param : tf.Variable
            Variable containing parameters to optimize
        grad : tf.Tensor
            Gradient tensor
        state : dict
            Any extra states required by the optimizer. Keys are strings;
            values are tf.Variable() tracked by add_slot/get_slot

        Returns
        -------
        (tf.Variable, dict)
            [0] : updated parameters
            [1] : updated state variables (same format as `state`)
        """
        raise NotImplementedError()

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
    def meta_loss(self, problem, weights):
        """Get meta training loss

        The caller is responsible for setting the initial values of the problem
        parameters, which are owned by `problem`.

        By decorating as @tf.function, the for loop should be wrapped into
        a tf.while_loop. See `https://www.tensorflow.org/guide/function`.

        Parameters
        ----------
        problem : tbd
            Optimizee module.
        weights : tf.Tensor
            Tensor specifying loss weights. The dimensionality specifies the
            number of unrolls. For example, [1 ... 1] indicates total loss,
            while [1/d ... 1/d] indicates mean loss and [0 ... 0 1] final loss.
        """

        loss = 0

        if self.obj_train_max_multiplier > 0:
            init_obj = problem.objective()
            max_obj = (
                (self.obj_train_max_multiplier - 1) * tf.abs(init_obj)
                + init_obj)

        # Create new slots
        # Should reset values if they already exist
        # Should also initialize state
        self._create_slots(problem.trainable_variables)

        # Size of weights determines unroll length
        for weight in tf.unstack(weights):

            # cond2: objective is still finite
            if not tf.math.is_finite(loss):
                break

            current_obj = problem.objective()

            # cond3: objective is a reasonable multiplier of the original
            if self.obj_train_max_multiplier > 0 and current_obj > max_obj:
                break

            # call this optimizer on the problem
            # outside in the meta-training loop, we will call
            # minimize(meta_loss, self.trainable_variables)

            # this calls self._compute_update via self._apply_dense
            self.minimize(current_obj, problem.trainable_variables)

            # Add to loss
            loss += self._scale_objective(current_obj)

        # @tf.function should compile this down as per tensorflow 2 best
        # practices
        return loss