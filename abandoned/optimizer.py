import tensorflow as tf
from tensorflow.keras import optimizers


def _var_key(var):
    """Key for representing a primary variable, for looking up slots.
    In graph mode the name is derived from the var shared name.
    In eager mode the name is derived from the var unique id.
    If distribution strategy exists, get the primary variable first.
    Args:
        var: the variable.
    Returns:
        the unique name of the variable.
    """

    # pylint: disable=protected-access
    # Get the distributed variable if it exists.
    if hasattr(var, "_distributed_container"):
        var = var._distributed_container()
    if var._in_graph_mode:
        return var._shared_name
    return var._unique_id


class LearnedOptimizer(optimizers.Optimizer):
    def __init__(self, policy):
        super(LearnedOptimizer, self).__init__()
        self.policy = policy

    def _resource_apply_dense(self, grad, var, apply_state):
        delta, states_new = self.policy.call(
            grad, self.hidden_states[_var_key(var)])

        self.hidden_states[_var_key(var)] = states_new
        var.assign_add(delta)

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        raise NotImplementedError()

    def _create_slots(self, var_list):
        self.hidden_states = {
            _var_key(var): self.policy.get_initial_state(
                batch_size=tf.size(var))
            for var in var_list
        }

    def get_config(self):
        raise NotImplementedError()

    def meta_minimize(self, make_loss, unroll, learning_rate=0.01, **kwargs):
        loss = self.meta_loss(make_loss, unroll, **kwargs)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        step = optimizer.minimize(loss)
