"""Abstract loss per-step callbacks."""

import tensorflow as tf


class BaseStepCallback:
    """Abstract loss step callback.

    Parameters
    ----------
    unroll : int
        Unroll length; used to prepare buffers.
    n_teachers : int
        Number of teachers.
    """

    key_names = []
    use_mean = []

    def __init__(self, unroll, n_teachers):
        pass

    def on_step_end(self, index, current_obj, teacher_loss):
        """Called at the end of every step.

        Parameters
        ----------
        index : int
            Index of current step.
        current_obj : float
            Problem objective value at this step (mini-batch).
        teacher_loss : float[]
            Array of per-teacher losses.
        """
        pass

    def summarize(self):
        """Summarize collected information into a dictionary of tensors.

        Returns
        -------
        dict
            Summarized information, most likely using tf.TensorArray().stack()
            or tf.reduce_sum/mean.
        """
        return {}


def is_callback(x):
    """Check if ```x``` extends BaseStepCallback."""
    return isinstance(x, BaseStepCallback)


class WhichTeacherCallback:
    """Callback to track which teacher is used at each iteration."""

    key_names = ["teacher_counts"]

    def __init__(self, unroll, n_teachers):
        self.indices = tf.TensorArray(tf.uint8, size=unroll)
        self.n_teachers = n_teachers

    def on_step_end(self, index, current_obj, teacher_loss):
        """Log argmax teacher loss."""
        self.indices.write(
            index, tf.math.argmax(teacher_loss, output_type=tf.uint8))

    def summarize(self):
        """Aggregate teacher argmax into a single array of counts."""
        stacked = self.indices.stack()
        return {"teacher_counts": [
            tf.reduce_sum(tf.equal(stacked, i))
            for i in range(self.n_teachers)
        ]}
