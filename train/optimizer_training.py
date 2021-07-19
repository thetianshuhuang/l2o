"""Learned Optimizer Training Class."""

import tensorflow as tf
import json
import os

from .loss_mixins import LossMixin
from .step_mixins import StepMixin
from .train_mixins import TrainingMixin
from .warmup_mixins import WarmupMixin

from . import gradient_clipping as gradient_clipping_module
from . import step_callbacks as step_callbacks_module
from .step_callbacks import BaseStepCallback, is_callback

from l2o import deserialize


class OptimizerTraining(LossMixin, StepMixin, TrainingMixin, WarmupMixin):
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
    do_teacher_parameter_scale : bool
        Controls whether the teachers should use scaled or unscaled gradients.
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
    huber_delta : float
        Delta parameter for huber loss used for imitation loss (applied
        parameterwise). If <0, ordinary l2 loss is used instead.
    gradient_clipping : dict
        Gradient clipping configuration.
    epsilon : float
        Epsilon value.
    step_callbacks : str[] or BaseStepCallback[]
        List of callbacks called at the end of every inner step; can be used to
        log inner step data. Does nothing by default.
        Behavior by type:
        - str : Uses corresponding class from train/step_callbacks.
        - BaseStepCallback : Uses ``step_callback``.
    mean_stats : str[]
        Statistics to reduce with reduce_mean and log in summary.csv.
    stack_stats : str[]
        Statistics to stack (np.stack) and save to a .npz file.
    pbar_values : str[]
        List of values to monitor in progress bar.
    """

    def __init__(
            self, network, optimizer, name="OptimizerTraining",
            use_log_objective=True, scale_objective=False,
            parameter_scale_spread=3.0, loss_reduce=tf.math.reduce_max,
            do_teacher_parameter_scale=True,
            il_mode='switch', unroll_weight="sum", teachers=[],
            obj_train_max_multiplier=-1, huber_delta=-1,
            gradient_clipping={
                "class_name": "SimpleGC", "config": {"clip_value": -1}},
            epsilon=1e-10, step_callbacks=[],
            pbar_values=["meta_loss", "imitation_loss"],
            mean_stats=["meta_loss", "imitation_loss"],
            stack_stats=[]):

        # Core
        self.name = name
        self.network = network
        self.optimizer = deserialize.optimizer(optimizer)

        # Checkpoints
        self.checkpoint = tf.train.Checkpoint(
            network=self.network, optimizer=self.optimizer)

        # Scaling & Transformation
        self.use_log_objective = use_log_objective
        self.scale_objective = scale_objective
        self.parameter_scale_spread = parameter_scale_spread
        self.do_teacher_parameter_scale = do_teacher_parameter_scale

        # Loss computation
        self.loss_reduce = deserialize.generic(
            loss_reduce, tf.math, pass_cond=callable,
            message="reduce function", default=tf.math.reduce_max)
        self.il_mode = il_mode
        self.unroll_weight = deserialize.weights(unroll_weight)
        self.teachers = [deserialize.policy(t) for t in teachers]

        # Numerical stability
        self.obj_train_max_multiplier = obj_train_max_multiplier
        self.huber_delta = huber_delta
        self.epsilon = epsilon
        self.gradient_clipping = deserialize.generic(
            gradient_clipping["class_name"], gradient_clipping_module,
            message="gradient clipping")(**gradient_clipping["config"])

        # Tracking
        self.step_callbacks = [
            deserialize.generic(
                cb, step_callbacks_module, pass_cond=is_callback,
                message="inner step callback", default=BaseStepCallback
            )(self) for cb in step_callbacks
        ]
        self.mean_stats = mean_stats
        self.stack_stats = stack_stats
        self.pbar_values = pbar_values

    def __str__(self):
        """As string -> <TrainableOptimizerName:NetworkName>."""
        return "<{}:{}>".format(self.name, self.network.name)
