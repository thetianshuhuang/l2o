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
    kwargs : dict
        Passed to ``init_layers``.
    """

    default_name = "LearnedOptimizer"

    def __init__(self, name=None, distribute=None, train=True, **kwargs):

        if name is None:
            name = self.default_name
        super().__init__(name)

        self.train = train
        self.config = kwargs

        if distribute is None:
            distribute = tf.distribute.get_strategy()
        with distribute.scope():
            self.init_layers(**kwargs)

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
