"""Base API documentation (for tools to pull parent docstrings)"""

import tensorflow as tf


class BaseCoordinateWiseNetwork(tf.keras.Model):

    def call(self, param, inputs, states):
        """Network call override (handled by tf.keras.Model)

        Parameters
        ----------
        param : tf.Variable
            Corresponding input variable.
        inputs : tf.Tensor
            Inputs; should be gradients.
        states : object
            Nested structure containing current hidden states; encoded by
            .get_initial_state

        Returns
        -------
        (tf.Tensor, object)
            [0] : Output; gradient delta
            [1] : New state

        Notes
        -----
        Rules are as of TF 2.3.0-RC1.
        (1) ``inputs`` may have undefined dimensions due to gradients not
            yet being connected to the graph yet. The dimension of ``param``
            should be used instead.
        (2) ``states`` MUST NOT be modified since this will be wrapped in
            a @tf.function. Create a new dictionary, and write updates to that
            dictionary.
        (3) The output ``states_new`` must have the EXACT same structure as the
            input structure. This means that tuples and lists cannot be used
            interchangeably. The starting value will be the structure returned
            by ``get_initial_state``.
        """
        raise NotImplementedError()

    def get_initial_state(self, var):
        """Get initial model state as a dictionary

        Parameters
        ----------
        var : tf.Variable
            Variable to create initial states for.

        Returns
        -------
        object
            Nested structure containing state information.

        Notes
        -----
        Rules are as of TF 2.3.0-RC1.
        (1) The state should consist only of ``tf.Tensor``s and cannot contain
            ``tf.Variable``. This is because variable assignments stop
            gradients. The ``TrainableOptimizer`` will wrap the state with
            ``tf.Variable`` during evaluation, but keep them as ``tf.Tensor``
            during training.
        """
        raise NotImplementedError()

    def call_global(self, states, global_state):
        """No action.

        Due to a tensorflow bug that attempts to convert parameters inside
        nested structures that are None to tf.Tensor, returns "0." instead of
        ``None`` as it should be.
        """
        return 0.

    def get_initial_state_global(self):
        """No global state."""
        return 0.


class BaseHierarchicalNetwork(tf.keras.Model):

    def call(self, param, grads, states, global_state):
        """Call function for parameter and tensor RNN updates

        The (param, grads, states) triple encodes the current optimization
        state for a single tensor. The call method should run parameter updates
        on the entire tensor as a batch, and tensor updates with a batch size
        of 1.
        If any per-tensor inputs are required for the global RNN, they should
        be prepared here.

        Parameters
        ----------
        param : tf.Variable
            Corresponding input variable.
        inputs : tf.Tensor
            Inputs; should be gradients.
        states : object
            Nested structure containing current states for parameter and tensor
            RNN.

        Returns
        -------
        (tf.Tensor, object)
            [0] : Output; gradient delta
            [1] : New state

        Notes
        -----
        The same rules as BaseCoordinateWiseNetwork should be followed.
        """

        raise NotImplementedError()

    def get_initial_state(self, var):
        """Get initial model state as a dictionary

        Parameters
        ----------
        var : tf.Variable
            Variable to create initial states for.

        Returns
        -------
        object
            Nested structure containing state information.

        Notes
        -----
        Rules are as of TF 2.3.0-RC1.
        (1) The state should consist only of ``tf.Tensor``s and cannot contain
            ``tf.Variable``. This is because variable assignments stop
            gradients. The ``TrainableOptimizer`` will wrap the state with
            ``tf.Variable`` during evaluation, but keep them as ``tf.Tensor``
            during training.
        """
        raise NotImplementedError()

    def call_global(self, states, global_state):
        """Call function for global RNN update

        Parameters
        ----------
        states : object[]
            List of nested structures corresponding to the optimizer states
            for all variables
        global_state : object
            Nested structure containing the global state

        Returns
        -------
        object
            New global state
        """
        raise NotImplementedError()

    def get_initial_state_global(self):
        """Initialize global hidden state

        Returns
        -------
        object
            Initial global hidden state
        """
        raise NotImplementedError()
