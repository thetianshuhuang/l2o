"""Multi-GPU computing methods."""
import tensorflow as tf


def distributed(f):
    """Decorator to invoke strategy variable mirroring."""
    def wrapped(self, *args, **kwargs):
        with self.distribute_strategy.scope():
            return f(self, *args, **kwargs)
    return wrapped


class DistributeMixin:
    """
    """

    def distributed_step(self, step_function, batch):
        """Run a distributed optimization step.

        Parameters
        ----------
        step_function : callable(obj) -> tf.Tensor, tf.Tensor[]
            Callable that computes loss and gradients from a data batch.
        """
        loss, grads = self.distribute_strategy.run(callable)
        reduce_op = tf.distribute.ReduceOp.SUM
        return (
            self.distribute_strategy.reduce(reduce_op, loss, axis=0),
            self.distribute_strategy.reduce(reduce_op, grads, axis=0)
        )
