"""Learned Optimizer Training Class."""

import tensorflow as tf
import json
import os

from .loss_mixins import LossMixin
from .step_mixins import StepMixin
from .train_mixins import TrainingMixin

from . import step_callbacks
from .step_callbacks import BaseStepCallback, is_callback

from l2o import deserialize


class OptimizerTraining(LossMixin, StepMixin, TrainingMixin):
    """Learned Optimizer Training Class.

    Parameters
    ----------
    network : tf.keras.Model
        Model containing the necessary call methods to operate the optimizer.
    optimizer : tf.keras.optimizers.Optimizer or str or dict
        Optimizer to train learned optimizer with; initialized with
        tf.keras.optimizers.get to support str and dict formats.

    Keyword Args
    ------------
    name : str
        Optimizer training name
    use_log_objective : bool
        Whether this optimizer uses the logarithm of the objective when
        computing the loss
    scale_objective : bool
        Whether the loss should be scaled by the initial value.
    parameter_scale_spread : float
        Each parameter is randomly scaled by a factor sampled from a
        log uniform distribution exp(Unif([-L, L])). If the spread is 0,
        this is equivalent to a constant scale of 1.
    loss_reduce : str or Callable (float[]) -> float or None
        Imitation learning multi-teacher loss strategy. Suggested:
        - ``tf.math.reduce_mean``: classic multi-teacher mean loss.
        - ``tf.math.reduce_max``: minimax loss.
        Behavior by type:
        - str : Uses ``tf.math.<loss_reduce>``.
        - Callable (float[]) -> float : uses ``loss_reduce``.
        - None : Uses ``tf.math.reduce_max``.
    il_mode : str
        Designates the imitation learning mode. Possible options:
        - 'switch': loss = IL w.p. p_teacher, ML w.p. 1 - p_teacher
        - 'sum': loss = p_teacher * IL + ML
    unroll_weight : str or Callable(int, int) -> float
        Callable specifying loss weights for each unroll iteration. If ``str``,
        can be the following options:
        - 'sum': total loss; f(i, n) -> 1
        - 'mean': mean loss; f(i, n) -> 1/n
        - 'final': final loss; f(i, n) -> 1_(i == n - 1)
    teachers : (str or dict or tf.keras.optimizers.Optimizer)[]
        List of optimizers to train against. Each teacher is deserialized by
        tf.keras.optimizers.get extended to tensorflow_addons if available; see
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/get.
    obj_train_max_multiplier : float
        The maximum multiplier for the increase in the objective before
        meta-training is stopped. If <= 0, meta-training is not stopped
        early.
    epsilon : float
        Epsilon value.
    step_callback : str or BaseStepCallback or None
        Called at the end of every inner step; can be used to log inner step
        data. Does nothing by default.
        Behavior by type:
        - str : Uses corresponding class from train/step_callbacks.
        - BaseStepCallback : Uses ``step_callback``.
        - None: Uses ``BaseStepCallback`` (does nothing).
    distribute : None or tf.distribute.Strategy
        Distributed training tensorflow strategy.
        If None, uses ``tf.distribute.get_strategy()``.
    """

    def __init__(
            self, network, optimizer, name="OptimizerTraining",
            use_log_objective=True, scale_objective=False,
            parameter_scale_spread=3.0, loss_reduce=tf.math.reduce_mean,
            il_mode='switch', unroll_weight="sum", teachers=[],
            obj_train_max_multiplier=-1, epsilon=1e-10, step_callback=None,
            distribute=None):

        # Core
        if distribute is None:
            distribute = tf.distribute.get_strategy()
        self.distribute = distribute

        self.name = name
        self.network = network
        self.optimizer = deserialize.optimizer(optimizer)

        # Checkpoints
        self.checkpoints = {
            "network": tf.train.Checkpoint(network=self.network),
            "optimizer": tf.train.Checkpoint(optimizer=self.optimizer)
        }

        # Scaling & Transformation
        self.use_log_objective = use_log_objective
        self.scale_objective = scale_objective
        self.parameter_scale_spread = parameter_scale_spread

        # Loss computation
        self.loss_reduce = deserialize.generic(
            loss_reduce, tf.math, pass_cond=callable,
            message="reduce function", default=tf.math.reduce_max)
        self.il_mode = il_mode
        self.unroll_weight = deserialize.weights(unroll_weight)
        self.teachers = [deserialize.optimizer(t) for t in teachers]

        # Numerical stability
        self.obj_train_max_multiplier = obj_train_max_multiplier
        self.epsilon = epsilon

        # Tracking
        self.step_callback = deserialize.generic(
            step_callback, step_callbacks, pass_cond=is_callback,
            message="inner step callback", default=BaseStepCallback)
        self.tracked_statistics = (
            ["imitation", "meta"] + self.step_callback.key_names)
        self.scalar_statistics = (
            ["imitation", "meta"] + self.step_callback.use_mean)

    def __str__(self):
        """As string -> <TrainableOptimizerName:NetworkName>."""
        return "<{}:{}>".format(self.name, self.network.name)

    def save_state(self, path):
        """Save learner and optimizer state.

        Parameters
        ----------
        path : str
            Directory to save to. Will create new if ``path`` does not exist.
        """
        os.makedirs(path, exist_ok=True)

        # Network config (use to initialize network when loading)
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(self.network.get_config(), f)

        # Weights
        for prefix, checkpoint in self.checkpoints.items():
            checkpoint.write(os.path.join(path, prefix))

        print("Saved training state: {}  -->  {}".format(str(self), path))

    def load_state(self, path):
        """Load learner and optimzier state.

        Parameters
        ----------
        path : str
            Directory to load from.
        """
        for prefix, checkpoint in self.checkpoints.items():
            checkpoint.read(os.path.join(path, prefix))
        print("Loaded training state: {}  -->  {}".format(path, str(self)))
