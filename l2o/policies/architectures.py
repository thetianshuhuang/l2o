"""Base API documentation (for tools to pull parent docstrings)."""

import tensorflow as tf
from l2o.optimizer import CoordinateWiseOptimizer, HierarchicalOptimizer


class BaseLearnToOptimizePolicy(tf.keras.Model):
    """Base L2O Class.

    Keyword Args
    ------------
    name : str or None
        Defaults to name specified by the ``default_name`` attribute.
    distribute : None or tf.distribute.Strategy
        Distributed training tensorflow strategy.
    train : bool
        Set to True when training, and False during evaluation. Used for
        policies with gradient estimation (i.e. gumbel softmax); can be ignored
        by most policies.
    debug : bool
        If True, sets a debug flag that indicates to optimizers they should log
        debug information in their optimizer states.
    weights_file : str or None
        Optional filepath to load optimizer network weights from.
    kwargs : dict
        Passed to ``init_layers``.
    """

    default_name = "LearnedOptimizer"

    def __init__(
            self, name=None, distribute=None, train=True, debug=False,
            weights_file=None, **kwargs):

        if name is None:
            name = self.default_name
        super().__init__(name)

        self.train = train
        self.debug = debug
        self.config = kwargs

        if distribute is None:
            distribute = tf.distribute.get_strategy()
        with distribute.scope():
            self.init_layers(**kwargs)

        if weights_file is not None:
            self.load_weights(weights_file)

    def load_weights(self, file):
        """Load saved weights from file."""
        tf.train.Checkpoint(network=self).read(file)

    def init_layers(self, **kwargs):
        """Initialize layers."""
        raise NotImplementedError()

    def get_config(self):
        """Get network config."""
        return self.config

    def call(self, param, inputs, states, global_state):
        """Network call override (handled by tf.keras.Model).

        Parameters
        ----------
        param : tf.Variable
            Corresponding input variable.
        inputs : tf.Tensor
            Inputs; should be gradients.
        states : object
            Nested structure containing current hidden states; encoded by
            .get_initial_state
        global_state : object
            Nested structure containing current global hidden state; can be
            empty.

        Returns
        -------
        (tf.Tensor, object)
            [0] : Output; gradient delta
            [1] : New state

        Notes
        -----
        Rules are as of TF 2.3.0-RC1.
        (1) ``inputs`` may have undefined dimensions due to gradients not
            being connected to the graph yet. The dimension of ``param``
            should be used instead.
        (2) ``states`` MUST NOT be modified since this will be wrapped in
            a @tf.function. Create a new dictionary, and write updates to that
            dictionary.
        (3) The output ``states_new`` must have the EXACT same structure as the
            input structure. This means that tuples and lists cannot be used
            interchangeably. The starting value will be the structure returned
            by ``get_initial_state``.
        (4) Due to straange behavior when returning python objects such as
            constant int, None, etc, return tf.constant(0.) when the state is
            empty.
        """
        raise NotImplementedError()

    def get_initial_state(self, var):
        """Get initial model state as a dictionary.

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
        """By default, perform no action.

        Due to a tensorflow bug that attempts to convert parameters inside
        nested structures that are None to tf.Tensor, returns ``0.`` instead of
        ``None`` as it should be.
        """
        return tf.constant(0.)

    def get_initial_state_global(self):
        """By default, there is no global state.

        See ```call_global``` for reason why we return 0.
        """
        return tf.constant(0.)

    def warmup_mask(self, state, new_state, in_warmup):
        """Mask state when in warmup to disable a portion of the update.

        Must be implemented with tf.cond due to @tf.function in update.

        Parameters
        ----------
        state : object
            Nested structure containing previous optimizer state.
        new_state : object
            Nested structure containing new optimizer state; can be assigned
            conditionally based on in_warmup.
        in_warmup : tf.bool
            Tensorflow bool indicating whether the optimizer is in warmup.

        Returns
        -------
        object
            New optimizer state with masked attributes
        """
        return new_state

    def debug(self, param, states):
        """Get debug information.

        Parameters
        ----------
        param : tf.Tensor
            Parameter variable. This should not be modified.
        states : object
            Nested structure of optimizer states, including debug.

        Returns
        -------
        dict
            Debug information. Empty unless implemented by the policy.
        """
        return {}

    def debug_global(self, global_state):
        """Get global debug information.

        Parameters
        ----------
        global_state : object
            Nested structure of global optimizer state.

        Returns
        -------
        dict
            Debug information. Empty unless implemented by the policy.
        """
        return {}

    def debug_summarize(self, params, debug_states, debug_global):
        """Summarize debug information.

        Parameters
        ----------
        params : tf.Variable[]
            List of problem variables to fetch summary for.
        debug_states : dict[]
            List of debug data for each variable.
        debug_global : dict
            Global debug information.
        """
        return {}


class BaseCoordinateWisePolicy(BaseLearnToOptimizePolicy):
    """Base Class for CoordinateWise L2O Policies."""

    architecture = CoordinateWiseOptimizer


class BaseHierarchicalPolicy(BaseLearnToOptimizePolicy):
    """Base Class for Hierarchical L2O Policies."""

    architecture = HierarchicalOptimizer

    def call_global(self, states, global_state):
        """Call function for global RNN update.

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
        """Initialize global hidden state.

        Returns
        -------
        object
            Initial global hidden state
        """
        raise NotImplementedError()
