import tensorflow as tf
import numpy as np
import collections

from tensorflow.keras.utils import Progbar

from .utils import reset_optimizer


MetaIteration = collections.namedtuple(
    "MetaIteration", [
        "problem", "optimizer",
        "unroll_len", "unroll_weights",
        "teachers", "imitation_optimizer", "strategy", "p_teacher",
        "validation"
    ])


_TrainingResults = collections.namedtuple("TrainingResults", ["loss", "mode"])


class TrainingResults(_TrainingResults):
    """Meta Iteration training results

    Attributes
    ----------
    loss : np.array(dtype=float32)[]
        Arrays of loss values; each epoch has its own array. For full batch
        training, loss has length 1, and all repeats are in the same np.array.
    mode : np.array(dtype=bool)[]
        Arrays of bool indicating whether imitation learning (True) or
        meta learning (False) was used for that iteration.
    """
    pass


builtin_weights = {
    "sum": lambda n: tf.ones([n]),
    "mean": lambda n: tf.ones([n]) / tf.cast(n, tf.float32)
}


class TrainingMixin:

    def _regen_teacher_vars(self, meta):
        """Helper function to force teacher optimizers to generate hidden
        state variables.

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

    def _make_cf(
            self, meta, weights, data, unroll, params=None, states=None,
            global_state=None, is_batched=False):
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

        kwargs = dict(
            params=params, states=states, global_state=global_state,
            unroll=unroll, problem=meta.problem, is_batched=is_batched)

        # P(meta learning) > 0
        if meta.p_teacher < 1:
            cf_meta = self.meta_loss.get_concrete_function(
                weights, data, noise_stddev=meta.problem.noise_stddev,
                **kwargs)
        else:
            cf_meta = None
        # Teachers are not empty and P(imitation learning) > 0
        if len(meta.teachers) > 0 and meta.p_teacher > 0:
            cf_imitation = self.imitation_loss.get_concrete_function(
                weights, data, teachers=meta.teachers,
                strategy=meta.strategy, **kwargs)
        else:
            cf_imitation = None

        return cf_meta, cf_imitation

    def _meta_step(
            self, meta, concrete_loss, weights, data,
            params=None, states=None, global_state=None):
        """Helper function to run for a single step."""

        cf_meta, cf_imitation = concrete_loss
        # Only imitation learning or only meta learning
        if cf_meta is None:
            is_imitation = True
        elif cf_imitation is None:
            is_imitation = False
        # Randomly select meta or imitation learning
        else:
            is_imitation = np.random.uniform(0, 1) > 0.5

        opt = meta.imitation_optimizer if is_imitation else meta.optimizer
        _loss = cf_imitation if is_imitation else cf_meta

        if meta.validation:
            # Validation mode -> only compute loss, no gradients
            loss, params, states, global_state = _loss(
                weights, data,
                params=params, states=states, global_state=global_state)
        else:
            # Specify trainable_variables specifically for efficiency
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.network.trainable_variables)
                # Other arugments of ``concrete_loss`` are bound and do not
                # need to be passed.
                loss, params, states, global_state = _loss(
                    weights, data,
                    params=params, states=states, global_state=global_state)
            # Standard apply_gradient paradigam
            # Used instead of ``optimizer.minimize`` to expose the current loss
            grads = tape.gradient(loss, self.network.trainable_variables)
            opt.apply_gradients(zip(grads, self.network.trainable_variables))

        return loss, is_imitation, params, states, global_state

    def _train_full(self, meta, repeat=1):
        """Full batch training.

        Parameters
        ----------
        meta : MetaIteration
            Current metaiteration parameters. See docstring.

        Keyword Args
        ------------
        repeat : int
            Number of times to repeat. reset() is used for computational
            efficiency (the loss graph is not rebuilt between repeats).

        Returns
        -------
        TrainingResults
            Loss and mode logging for this meta-iteration
        """
        pbar = Progbar(repeat, unit_name='step')

        # Note: concrete_loss is a tuple of concrete functions
        # [0]: meta_loss; [1]: imitation_loss
        concrete_loss = None
        unroll = meta.unroll_len()
        weights = meta.unroll_weights(unroll)

        # Logging
        losses = np.zeros(repeat, dtype=np.float32)
        modes = np.zeros(repeat, dtype=np.bool)

        for i in range(repeat):

            data = meta.problem.get_internal()

            self.reset()
            meta.problem.reset(internal=data)
            # State (i.e. momentum) needs to be reset
            for t in meta.teachers:
                reset_optimizer(t)

            # Only create concrete loss on first iteration
            if concrete_loss is None:
                concrete_loss = self._make_cf(meta, weights, data, unroll)

            # Ignore all param & state arguments
            loss, mode, _, _, _ = self._meta_step(
                meta, concrete_loss, weights, data)

            # Logging
            pbar.add(1, values=[("loss", loss)])
            losses[i] = loss.numpy()
            modes[i] = mode

        return TrainingResults(loss=[losses], mode=[modes])

    def _train_batch(self, meta, epochs=1, persistent=False):
        """Minibatch training.

        Parameters
        ----------
        meta : MetaIteration
            Current metaiteration parameters. See docstring.

        Keyword Args
        ------------
        epochs : int
            Number of epochs to run for.
        persistent : bool
            If True, batch training keeps a persistent optimizer and optimizee
            state across iteration trajectories. If False, the optimizer state
            is reset after every iteration.

        Returns
        -------
        TrainingResults
            Loss and mode logging for this meta-iteration
        """
        if persistent:
            params, states, global_state = self._get_state(
                meta.problem, params=None, states=None, global_state=None)
        else:
            params = meta.problem.get_parameters()
            states = None
            global_state = None

        # Logging
        losses = []
        modes = []

        # Generate unrolls (to draw progress bar)
        unrolls = [meta.unroll_len() for _ in range(epochs)]

        # Progress bar
        sizes = [meta.problem.size(unroll) for unroll in unrolls]
        size = sum(sizes)
        pbar = Progbar(size, unit_name='step')

        # See docstring for why this is necessary
        self._regen_teacher_vars(meta)

        for i, (size, unroll) in enumerate(zip(sizes, unrolls)):

            # Get unroll weights; concrete_loss must be regenerated due
            # to the change to ``unroll``.
            weights = meta.unroll_weights(unroll)
            dataset = meta.problem.get_dataset(unroll)
            concrete_loss = None

            # Logging
            losses.append(np.zeros(size, dtype=np.float32))
            modes.append(np.zeros(size, dtype=np.bool))

            for j, batch in enumerate(dataset):
                # State (i.e. momentum) needs to be reset
                if not persistent:
                    self.reset()
                # Sync with student
                meta.problem.reset(values=params)
                # Reset teachers
                if not persistent:
                    for t in meta.teachers:
                        reset_optimizer(t)

                # Data dimensions are ``[unroll, batch] + [data]``
                batch_stacked = [
                    tf.stack(tf.split(dim, num_or_size_splits=unroll))
                    for dim in batch]

                # Only create concrete loss on first iteration
                if concrete_loss is None:
                    concrete_loss = self._make_cf(
                        meta, weights, batch_stacked, unroll, params=params,
                        states=states, global_state=global_state,
                        is_batched=True)

                # The actual step
                loss, mode, params, states, global_state = self._meta_step(
                    meta, concrete_loss, weights, batch_stacked,
                    params=params, states=states, global_state=global_state)

                # not persistent -> discard state, global_state
                if not persistent:
                    states = None
                    global_state = None

                # Logging
                pbar.add(1, values=[("loss", loss)])
                losses[i][j] = loss.numpy()
                modes[i][j] = mode

        return TrainingResults(loss=losses, mode=modes)

    def train(
            self, problems, optimizer,
            unroll_len=lambda: 20, unroll_weights="sum",
            teachers=[], imitation_optimizer=None,
            strategy="mean", p_teacher=0,
            epochs=1, repeat=1, persistent=False, validation=False):
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
        imitation_optimizer : tf.keras.optimizers.Optimizer
            Separate optimizer to use on imitation loss updates if present.
            This may benefit optimization by keeping separate optimizer
            states for imitation and meta learning, as those losses may have
            vastly different gradient magnitudes.
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
        repeat : int
            Number of repetitions to run using the same graph if full batched.
        persistent : bool
            If True, batch training keeps a persistent optimizer and optimizee
            state across iteration trajectories. If False, the optimizer state
            is reset after every iteration.
        validation : bool
            If True, runs in validation mode (does not perform any parameter
            updates)

        Returns
        -------
        TrainingResults[]
            Logged loss and training modes for all problems; arranged in the
            same order as ``problems``.
        """
        # No imitation optimizer -> use same optimizer for both
        if imitation_optimizer is None:
            imitation_optimizer = optimizer

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

            meta = MetaIteration(
                problem, optimizer, unroll_len, unroll_weights, teachers,
                imitation_optimizer, strategy, p_teacher, validation)

            if hasattr(problem, "get_dataset"):
                results.append(
                    self._train_batch(
                        meta, epochs=epochs, persistent=persistent))
            elif hasattr(problem, "get_internal"):
                results.append(
                    self._train_full(meta, repeat=repeat))
            else:
                raise TypeError(
                    "Problem must be able to either get_dataset() or"
                    + "get_internal().")

        return results
