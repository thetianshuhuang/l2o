"""L2O Optimization Step."""
import tensorflow as tf


class StepMixin:
    """Optimization Step Mixin."""

    def _base_step(self, opt, callable):
        """Run a single step.

        In addition to the standard GradientTape -> gradient -> apply_gradients
        paradigam, additional logic is included to detect 'first time'
        execution of the keras model ``self.network``.

        This is because ``trainable_variables`` is not defined beforehand,
        so ``watch_accessed_variables`` needs to be set to True in order to
        capture them as they are created. However, this is still not the
        default in order to maintain efficiency by ignoring non-trainable
        variables.
        """
        # trainable_variables not yet built -> capture all variables
        if len(self.network.trainable_variables) <= 2:
            with tf.GradientTape() as tape:
                results = callable()
        # trainable_variables built -> capture only learner variables
        else:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.network.trainable_variables)
                results = callable()

        # Gradients
        # Loss is results[0]
        grads = tape.gradient(results[0], self.network.trainable_variables)
        opt.apply_gradients(zip(grads, self.network.trainable_variables))

        return results

    @tf.function
    def abstract_step(
            self, weights, data, unroll_state, opt=None, **kwargs):
        """Wraps imitation_loss to compute meta-gradients inside graph mode.

        See ``abstract_loss`` for docstring and ``_base_step`` for internal
        mechanism.

        NOTE: the *args must be manually iterated below due to a
        tensorflow bug that causes an internal IndexError when turning this
        into a concrete function.
        """
        def loss_wrapper():
            return self.abstract_loss(weights, data, unroll_state, **kwargs)

        return self._base_step(opt, loss_wrapper)
