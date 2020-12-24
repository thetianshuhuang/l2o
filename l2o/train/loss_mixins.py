"""Inner training and outer loss computation."""

import collections
import tensorflow as tf

from .random_scaling import create_random_parameter_scaling
from .unroll_state import UnrollStateManager, state_distance


class LossMixin:
    """Inner Training and Outer Loss computation Mixin."""

    def _scale_meta_objective(self, objective, initial_obj):
        """Normalizes the objective based on the initial objective value."""
        if self.use_log_objective:
            return (
                tf.math.log(objective + self.epsilon)
                - tf.math.log(initial_obj + self.epsilon))
        else:
            return objective / (initial_obj + self.epsilon)

    def _imitation_objective(self, teacher_losses):
        """Compute IL objective from individual teacher losses."""
        if len(self.teachers) > 0:
            if self.use_log_objective:
                return tf.math.log(self.loss_reduce(teacher_losses))
            else:
                return self.loss_reduce(teacher_losses)
        # Manually check here since reduce_{} often returns NaN when empty
        else:
            return 0.0

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

    def abstract_loss(self, data, params, unroll=20, problem=None, seed=None):
        """Get abstract imitation learning and meta learning loss.

        Runs inner training in order to compute the abstract loss
        ```
        strategy(imitation_loss(teachers)) + meta_loss_weight * meta_loss().
        ```

        By decorating as a @tf.function, the for loop is wrapped into a
        tf.while_loop. See `https://www.tensorflow.org/guide/function`.

        The following rules must be followed:
        (1) No variable creation. Variable creation is only allowed in eager
            mode.
        (2) No ``tf.Variable``s may be assigned, since this stops gradients.
            This precludes the use of tf.keras.Model in training problems, as
            well as usage of the ``_create_slots`` system.
        (3) Loop variables must be homogenous. In other words, local objects
            containing tensors which are modified each iteration must ONLY
            contain tensors that are modified each iteration.
        (4) Lists and tuples are not compatible. Objects which start as lists
            must remain as lists.
        (5) As a corollary of (3) and (4), non-tensors (bound python objects)
            must be stored in separate objects. This leads to the awkward
            ``list(map(list, zip(*[])))`` list[] -> list[] paradigm.

        Parameters
        ----------
        data : tf.Tensor[]
            List of data tensors.
        params : tf.Tensor[]
            Initial problem parameter values.

        Keyword Args
        ------------
        unroll : int
            Number of unroll iterations
        problem : problems.Problem
            Training problem
        seed : int or None
            Seed to use for intializing parameters.

        Returns
        -------
        (tf.Tensor, tf.Tensor[], tf.Tensor{})
            [0] Meta loss.
            [1] Final problem parameters.
            [2] Summary statistics collected by self.step_callback if present.
        """
        # Make random scaling
        params_scale, transform = create_random_parameter_scaling(
            params, spread=self.parameter_scale_spread, seed=seed)
        # Make states: [trained policy, teacher policy #1, ...]
        policies = [self.network, *self.teachers]

        def mgr_constructor(p):
            return UnrollStateManager(
                p, problem.objective, transform, [True for _ in params_scale])

        policy_managers = [mgr_constructor(p) for p in policies]
        unroll_states = [p.create_state(params_scale) for p in policy_managers]

        meta_loss = 0.
        imitation_loss = 0.
        callback_states = [cb.get_state(unroll) for cb in self.step_callbacks]

        dataset = problem.prepare_batches(unroll, data)
        for i in tf.range(unroll):
            weight = self.unroll_weight(i, unroll)
            batch = problem.get_batch(dataset, i)

            # Advance by one step
            losses, unroll_states = list(map(list, zip(*[
                mgr.advance_state(st, batch)
                for st, mgr in zip(unroll_states, policy_managers)
            ])))

            # Scale objective
            if self.scale_objective:
                init_obj = 1.
            else:
                init_obj = problem.objective(params, batch)

            # Add meta loss
            if self._max_obj(init_obj, losses[0]):
                break
            meta_loss += (
                weight * self._scale_meta_objective(losses[0], init_obj))

            # Add imitation loss
            teacher_loss = [
                state_distance(unroll_states[0], s) for s in unroll_states[1:]]
            imitation_loss += weight * self._imitation_objective(teacher_loss)

            # Log optional statistics
            callback_states = [
                cb.on_step_end(st, i, losses[0], teacher_loss)
                for st, cb in zip(callback_states, self.step_callbacks)]

        params = transform(unroll_states[0].params)
        return meta_loss, imitation_loss, params, callback_states
