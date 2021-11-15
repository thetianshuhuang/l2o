"""Abstract loss per-step callbacks."""

import tensorflow as tf


class BaseStepCallback:
    """Abstract loss step callback.

    Parameters
    ----------
    parent : train.OptimizerTraining
        Parent training class.
    """

    keys = {}

    def __init__(self, parent):
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


class WhichTeacherCountCallback(BaseStepCallback):
    """Callback to track which teacher is used at each iteration."""

    def __init__(self, parent):
        self.n_teachers = len(parent.teachers)

    def get_state(self, unroll):
        """Get initial state."""
        return tf.zeros([self.n_teachers], dtype=tf.int32)

    def on_step_end(self, state, index, current_obj, teacher_loss):
        """Count argmax teacher loss."""
        indicator = tf.math.equal(
            tf.range(self.n_teachers),
            tf.math.argmax(teacher_loss, output_type=tf.dtypes.int32))
        return state + tf.cast(indicator, tf.int32)

    def summarize(self, state, distribute):
        """Generate summary."""
        reduced = distribute.reduce(
            tf.distribute.ReduceOp.SUM, state, axis=None)
        return {"teacher_counts": reduced}


class TeacherLossCallback(BaseStepCallback):
    """Callback to record teacher losses.

    NOTE: does not work in a distributed environment.
    """

    def __init__(self, parent):
        self.n_teachers = len(parent.teachers)

    def get_state(self, unroll):
        """Get initial state."""
        return tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    def on_step_end(self, state, index, current_obj, teacher_loss):
        """Count argmax teacher loss."""
        return state.write(index, teacher_loss)

    def summarize(self, state, distribute):
        """Generate summary."""
        return {"teacher_loss": state.stack()}
