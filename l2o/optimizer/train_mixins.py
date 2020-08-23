"""Optimizer Training."""
import tensorflow as tf
import numpy as np
import collections

from tensorflow.keras.utils import Progbar

from .utils import reset_optimizer


MetaIteration = collections.namedtuple(
    "MetaIteration", [
        "problem", "optimizer",
        "unroll_len", "weights",
        "teachers", "strategy", "p_teacher",
        "validation", "seed", "persistent", "parameter_scale_spread"
    ])


builtin_weights = {
    "sum": lambda n: tf.ones([n]),
    "mean": lambda n: tf.ones([n]) / tf.cast(n, tf.float32)
}


class Loss:
    """Helper object to hold training loss."""

    def __init__(self, *args):
        self.losses = {label: [] for label in args}

    def add(self, label, value):
        """Add loss."""
        self.losses[label].append(value)

    def get(self, label, mean=True):
        """Get loss."""
        if mean:
            if len(self.losses[label]) == 0:
                return 0
            else:
                return np.mean(self.losses[label])
        else:
            return self.losses[label]


class TrainingMixin:
    """Training Method Mixins for TrainableOptimizer."""

    def _regen_teacher_vars(self, meta):
        """Force teacher optimizers to generate hidden state variables.

        As of 2.3.0-rc2, I believe f.keras.optimizers.Optimizer has a
        compatibility issue with get_concrete_function. Using
        get_concrete_function triggers two traces, and sometimes causes issues
        on the second retrace with the optimizer trying to create variables.
        Therefore, this method forcibly generates hidden variables outside of
        the @tf.function loss functions to avoid this bug.
        """
        if len(meta.teachers) > 0:
            pairs = zip(meta.teachers, meta.problem.trainable_variables)
            for teacher, var_set in pairs:
                teacher._create_all_weights(var_set)

    def _make_cf(self, meta, data, unroll_state, is_batched=False):
        """Helper function to make a concrete function for the given params.

        A concrete function is a single graph generated by AutoGraph; see
        ``https://www.tensorflow.org/guide/concrete_function``.

        In general, the rules are (as of 2.3.0-rc1):
          - Nested structures (lists, dicts of Tensors) must maintain the same
            internal values and dimensions
          - Python objects are 'bound' and must be constant (i.e. id())
          - BUG: non-primitive python objects can only be passed during
            .get_concrete_function() and must not be passed again when called.
            This is because non-primitive objects are interpreted as
            ``UnknownArgument`` by tensorflow.
        """
        # Weights are just placeholders
        kwargs = dict(
            unroll=meta.unroll_len, problem=meta.problem,
            is_batched=is_batched, seed=meta.seed,
            meta_loss_weight=tf.constant(0.5),
            imitation_loss_weight=tf.constant(0.5),
            teachers=meta.teachers, strategy=meta.strategy,
            parameter_scale_spread=meta.parameter_scale_spread)
        args = (meta.weights, data, unroll_state)

        if meta.validation:
            return self.abstract_loss.get_concrete_function(*args, **kwargs)
        else:
            return self.abstract_step.get_concrete_function(
                *args, opt=meta.optimizer, **kwargs)

    def _meta_step(self, meta, concrete_step, data, unroll_state):
        """Helper function to run for a single step."""
        is_imitation = np.random.uniform(0, 1) < meta.p_teacher
        w_meta, w_imit = (0.0, 1.0) if is_imitation else (1.0, 0.0)

        loss, unroll_state = concrete_step(
            meta.weights, data, unroll_state,
            meta_loss_weight=tf.constant(w_meta),
            imitation_loss_weight=tf.constant(w_imit))
        return loss, unroll_state, is_imitation

    def _train_full(self, meta, repeat=1):
        """Full batch training.

        Parameters
        ----------
        meta : MetaIteration
            Current metaiteration parameters. See ``train`` docstring.

        Keyword Args
        ------------
        repeat : int
            Number of times to repeat.

        Returns
        -------
        float
            Mean training loss for this meta-iteration
        """
        # No states are persistent
        unroll_state = self._make_unroll_state(
            meta.problem, params=None, states=None, global_state=None,
            seed=meta.seed)
        concrete_step = None

        pbar = Progbar(repeat, unit_name='step')
        losses = Loss("meta", "imitation")

        for i in range(repeat):

            data = meta.problem.get_internal(seed=meta.seed)
            meta.problem.reset(internal=data)

            # State (i.e. momentum) needs to be reset
            for t in meta.teachers:
                reset_optimizer(t)

            # Only create concrete loss on first iteration
            if concrete_step is None:
                concrete_step = self._make_cf(meta, data, unroll_state)

            loss, unroll_state, is_imitation = self._meta_step(
                meta, concrete_step, data, unroll_state)

            pbar.add(1, values=[("loss", loss)])
            losses.add("imitation" if is_imitation else "meta", loss.numpy())

        return losses.get("imitation"), losses.get("meta")

    def _train_batch(
            self, meta, epochs=1, repeat=1, depth=1, persistent=False):
        """Minibatch training.

        Parameters
        ----------
        meta : MetaIteration
            Current metaiteration parameters. See ``train`` docstring.

        Keyword Args
        ------------
        epochs : int
            Number of epochs to run for.
        repeat : int
            Number of times to repeat. Will reset at the end of every repeat.
        persistent : bool
            Keeps a persistent optimizer?
        depth : int
            Optimization depth.

        Returns
        -------
        float
            Mean training loss for this meta-iteration
        """
        # concrete_step, unroll_state will be assigned on first iteration.
        concrete_step = None
        unroll_state = None

        # Single progress bar
        size = meta.problem.size(meta.unroll_len)
        pbar = Progbar(size * epochs * repeat, unit_name='step')
        losses = Loss("meta", "imitation")

        # See docstring for why this is necessary
        self._regen_teacher_vars(meta)

        # Generate seeds.
        # If None, keep None and let each component decide.
        # If present, generate a new seed for each epoch.
        if meta.seed is None:
            seeds = [None for _ in range(epochs * repeat)]
        else:
            np.random.seed(meta.seed)
            seeds = [
                np.random.randint(0, 0x80000000)
                for _ in range(epochs * repeat)]

        for i, seed in enumerate(seeds):
            # Get new state for each repeat
            if i % epochs == 0:
                unroll_state = self._make_unroll_state(
                    meta.problem, params=True, states=persistent,
                    global_state=persistent, seed=seed)
            # New dataset using seed for each epoch.
            dataset = meta.problem.get_dataset(meta.unroll_len, seed=seed)
            for j, batch in enumerate(dataset):
                # Every ``depth`` iterations, reset parameters
                if depth > 0 and (j + 1) % depth == 0:
                    unroll_state = self._reset_params(
                        unroll_state, meta.problem, seed=seed)

                # State (i.e. momentum) needs to be reset
                if not meta.persistent:
                    for t in meta.teachers:
                        reset_optimizer(t)
                # Sync with student
                meta.problem.reset(values=unroll_state.params)

                # Data dimensions are ``[unroll, batch] + [data]``
                batch_stacked = [
                    tf.stack(tf.split(dim, num_or_size_splits=meta.unroll_len))
                    for dim in batch]
                # Only create concrete loss on first iteration
                if concrete_step is None:
                    concrete_step = self._make_cf(
                        meta, batch_stacked, unroll_state, is_batched=True)

                # The actual step
                loss, unroll_state, is_imitation = self._meta_step(
                    meta, concrete_step, batch_stacked, unroll_state)

                pbar.add(1, values=[("loss", loss)])
                losses.add(
                    "imitation" if is_imitation else "meta", loss.numpy())

        return losses.get("imitation"), losses.get("meta")

    def train(
            self, problems, optimizer,
            unroll_len=lambda: 20, unroll_weights="sum",
            teachers=[], strategy="mean", p_teacher=0,
            epochs=1, depth=0, repeat=1, persistent=False,
            validation=False, seed=None, parameter_scale_spread=0.0):
        """Run meta-training.

        Parameters
        ----------
        problems : problem.ProblemSpec[]
            List of problem specifications to build and run
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer to use for meta optimization

        Keyword Args
        ------------
        unroll_len : Callable -> int
            Unroll size or callable that returns unroll size.
        unroll_weights : str or Callable(int) -> tf.Tensor
            Callable that generates unroll weights from an unroll size.
        teachers : tf.keras.optimizers.Optimizer[]
            If passed, runs imitation learning instead against ``teacher``.
        strategy : str or Callable (float[] -> float)
            Imitation learning multi-teacher loss strategy. Suggested:
              - "mean" or ``tf.math.reduce_mean``: classic mean loss.
              - "max" or ``tf.math.reduce_max``: minimax loss.
            Can also implement a custom multi-teacher strategy.
        p_teacher : float
            Probability of choosing imitation learning. Cannot be >0 if
            teachers is empty.
        epochs : int
            Number of epochs to run if batched
        depth : int
            Optimization depth, in meta-iterations, before the parameters
            should be reinitialized. If 0, is treated as infinity (no resets).
            A larger optimization depth allows more opportunities to train
            on more refined optimization. Only operates when not in persistent
            mode.
        repeat : int
            Number of repetitions to run using the same graph if full batched.
        persistent : bool
            If True, batch training keeps a persistent optimizer and optimizee
            state across iteration trajectories. If False, the optimizer state
            is reset after every iteration.
        validation : bool
            If True, runs in validation mode (does not perform any parameter
            updates)
        seed : int
            Random seed to use for model initializations. If None, no specific
            seed is used. Should be set to None to reduce overfitting during
            training, but fixed during validation.
        parameter_scale_spread : float
            Each parameter is randomly scaled by a factor sampled from a
            log uniform distribution exp(Unif([-L, L])). If the spread is 0,
            this is equivalent to a constant scale of 1.

        Returns
        -------
        float[][2]
            Logged imitation loss and meta loss for all problems;
            arranged in the same order as ``problems``.
        """
        results = []

        # Deserialize
        if type(unroll_weights) == str:
            try:
                unroll_weights = builtin_weights[unroll_weights]
            except KeyError:
                raise ValueError(
                    "Invalid unroll_weights: {}".format(unroll_weights))
        if type(strategy) == str:
            try:
                strategy = getattr(tf.math, "reduce_" + strategy)
            except AttributeError:
                raise ValueError(
                    "Invalid reduce strategy: {}".format("reduce_" + strategy))
        # tf.keras.optimizers.get will pass through if already a optimizer
        teachers = [tf.keras.optimizers.get(t) for t in teachers]

        for itr, spec in enumerate(problems):
            spec.print(itr)
            problem = spec.build(persistent=len(teachers))
            unroll = unroll_len()

            meta = MetaIteration(
                problem, optimizer, unroll, unroll_weights(unroll), teachers,
                strategy, p_teacher, validation, seed, persistent,
                parameter_scale_spread)

            if hasattr(problem, "get_dataset"):
                results.append(
                    self._train_batch(
                        meta, repeat=repeat, epochs=epochs, depth=depth))
            elif hasattr(problem, "get_internal"):
                results.append(
                    self._train_full(meta, repeat=repeat))
            else:
                raise TypeError(
                    "Problem must be able to either get_dataset() or"
                    + "get_internal().")

        return results
