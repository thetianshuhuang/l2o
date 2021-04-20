"""Hessian statistics computation."""

import tensorflow as tf
import collections

from tensorflow.keras.utils import Progbar
from .utils import make_seeds


MetaHessian = collections.namedtuple(
    "MetaHessian", ["unroll_len", "validation"])


class HessianMixin:
    """Hessian statistics computation mixin.

    Based on https://github.com/amirgholami/PyHessian; implements the ``Hv``
    mechanism described.
    """

    @tf.function
    def hessian_projection_step(self, data, states, scale, vs, **kwargs):
        """Single hessian projection step."""

        def _inner(data_, states_, scale_):
            """Distribute function.

            Since ``data`` and ``params`` contain per-replica tensor values,
            they must be explicitly passed as parameters; then, since
            ``kwargs`` only contains bound constants, they are captured by
            closure instead of passed through ``distribute.run``.
            """
            with tf.GradientTape(watch_accessed_variables=False) as otape:
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(self.network.trainable_variables)
                    loss, _, states_new, _ = self.abstract_loss(
                        data_, states_, scale_, **kwargs)

                grads = tape.gradient(loss, self.network.trainable_variables)
                grads_projected = [tf.reduce_sum(g * v) for g, v in zip(g, vs)]

            outer_grads = otape.gradient(
                grads_projected, self.network.trainable_variables)

            return states_new, outer_grads

        distribute = tf.distribute.get_strategy()
        states_new, Hv = distribute.run(_inner, args=(data, states, scale))

        Hv_reduced = [
            distribute.reduce(tf.distribute.ReduceOp.MEAN, g, axis=None)
            for g in outer_grads]

        return states_new, Hv_reduced

    def _hessian_projection(
            self, problem, unroll_len, depth=1, seed=None,
            epochs=1, warmup=0, warmup_rate=0.01):
        """Run a single round of hessian projection."""
        _meta = MetaHessian(unroll_len, "hessian")
        step = problem.get_step(_meta)
        warmup_step = None

        pbar = ProgBar(epochs * depth, unit_name='step')
        dataset = problem.get_dataset(
            unroll_len, epochs * (depth + warmup), seed=seed, load_all=True)

        Hv_acc = None
        tf.random.set_seed(seed)
        for i, batch in enumerate(dataset):
            # Reset params & states
            if i % (depth + meta.warmup) == 0:
                params = meta.problem.get_parameters(seed=seed)
                scale = [tf.ones(tf.shape(p)) for p in params]
                states = [create_state(self.network, params)]

            # Create concrete_step; done here to capture batch shape.
            args = (batch, states, scale)
            if step is None:
                step = self.make_concrete_step(meta, *args)
            if warmup_step is None and warmup > 0:
                warmup_step = self.make_warmup_concrete_step(meta, *args)

            # Warmup
            if i % (depth + warmup) < warmup:
                states = warmup_step(
                    *args, warmup_rate=tf.constant(warmup_rate))
            # The actual step
            else:
                states, Hv_new = step(*args)
                if Hv_acc is None:
                    Hv_acc = Hv_new
                else:
                    Hv_acc = [a + n for a, n in zip(Hv_acc, Hv_new)]
                pbar.add(1)

            # Dataset size doesn't always line up
            if i >= epochs * (depth + meta.warmup):
                break

        return [a / (epochs * depth) for a in Hv_acc]
