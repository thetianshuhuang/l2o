import tensorflow as tf
import time
import collections


def weights_sum(n):
    return tf.ones([n])


def weights_mean(n):
    return tf.ones([n]) / tf.cast(n, tf.float32)


_MetaIteration = collections.namedtuple(
    'MetaIteration', [
        'learner', 'teacher', 'problem', 'optimizer',
        'weights', 'unroll', 'noise_stddev',
        'epochs', 'idx'
    ])


class MetaIteration(_MetaIteration):
    """Meta iteration specifications.

    Attributes
    ----------
    learner : trainable_optimizer.TrainableOptimizer
        Optimizer train; owns the L2O algorithm to be trained.
    teacher : tf.keras.optimizers.Optimizer
        Optimier to imitate. None if using standard meta-learning.
    problem : problem.ProblemSpec
        Problem specification to train meta-epoch on. Must be built outside of
        a @tf.function before usage.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer to use for meta-optimization
    weights : callable (int -> tf.Tensor[unroll])
        Callable to create unroll weights from unroll length. Does not need
        to be constant.
    unroll : int
        Unroll length.
    noise_stddev : float
        Noise to add to problem gradients during meta-optimization.
    epochs : int
        Number of epochs to run meta-iteration problem for.
    idx : int
        Meta-iteration index for tracking
    """
    pass


def train_meta(itr):
    """Meta training on a single problem for a single meta-iteration

    Parameters
    ----------
    itr : MetaIteration
        Iteration to run
    """

    # Build problem
    problem = itr.problem.build()
    itr.problem.print(itr.idx)
    unroll_weights = itr.weights(itr.unroll)

    progress = tf.keras.utils.Progbar(
        None, unit_name='meta-iteration')

    # Prepare learner
    itr.learner._create_slots(problem.trainable_variables)

    for _ in range(itr.epochs):

        # Reset problem and rebatch
        # (both methods optionally implemented by problem)
        dataset = problem.get_dataset(itr.unroll)
        problem.reset()

        for batch in dataset:
            with tf.GradientTape() as tape:
                loss = itr.learner.meta_loss(
                    problem, unroll_weights, itr.unroll,
                    data=batch, noise_stddev=itr.noise_stddev)

            grads = tape.gradient(loss, itr.learner.trainable_variables)
            itr.optimizer.apply_gradients(
                zip(grads, itr.learner.trainable_variables))

            progress.add(1, values=("meta-loss", loss))


def train_imitation(itr):
    """Imitation learning on a single problem

    Parameters
    ----------
    itr : MetaIteration
        Iteration to run
    """

    # Build problem
    student_cpy = itr.problem.build()
    teacher_cpy = student_cpy.clone_problem()
    itr.problem.print(itr.idx)
    unroll_weights = itr.weights(itr.unroll)

    progress = tf.keras.utils.Progbar(
        None, unit_name='meta-iteration')

    # Prepare learner
    itr.learner._create_slots(student_cpy.trainable_variables)
    itr.teacher._create_slots(teacher_cpy.trainable_variables)

    for _ in range(itr.epochs):

        # Reset problem and rebatch
        # (both methods optionally implemented by problem)
        dataset = student_cpy.get_dataset(itr.unroll)
        student_cpy.reset(copy=teacher_cpy)

        for batch in dataset:
            with tf.GradientTape() as tape:
                loss = itr.learner.imitation_loss(
                    student_cpy, teacher_cpy, itr.teacher, unroll_weights,
                    itr.unroll, data=batch, noise_stddev=itr.noise_stddev)

            grads = tape.gradient(loss, itr.learner.trainable_variables)
            itr.optimizer.apply_gradients(
                zip(grads, itr.learner.trainable_variables))

            progress.add(1, values=("meta-loss", loss))


def train(
        learner, problems, optimizer,
        unroll=20, unroll_weights=weights_mean, teacher=None,
        epochs=1, noise_stddev=0.0):
    """Main Training Loop; a single meta-epoch.

    NOTE: this function cannot be converted using @tf.function / AutoGraph
    since problem.build() creates new variables.

    Parameters
    ----------
    learner : trainable_optimizer.TrainableOptimizer
        Optimizer to train
    problems : problem.ProblemSpec[]
        Problem specifications
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer for meta optimization

    Keyword Args
    ------------
    unroll : int
        Number of unroll iterations.
    unroll_weights : callable (int -> tf.Tensor[unroll])
        Function that creates a tensor encoding iteration weights for the
        truncated BPTT loss calculation
    teacher : tf.keras.optimizers.Optimizer
        Optimizer to emulate. If None, meta training loss is used. Otherwise,
        imitation learning is used.
    epochs : int
        Number of consecutive meta-iterations or epochs to repeat each problem
        for. The problem's ``reset`` method is called on each epoch.
    noise_stddev : tf.Tensor | float
        Normally distributed noise to add to optimizee gradients; use to
        simulate minibatch noise for full-batch problems.

    Returns
    -------
    float
        Time used during training; can be ignored.
    """

    start = time.time()

    for itr, problem in enumerate(problems):

        meta_itr = MetaIteration(
            learner=learner,
            teacher=teacher,
            problem=problem,
            optimizer=optimizer,
            weights=unroll_weights,
            unroll=tf.constant(unroll),
            noise_stddev=noise_stddev,
            epochs=epochs,
            idx=itr)

        if teacher is None:
            train_meta(meta_itr)
        else:
            train_imitation(meta_itr)

    return time.time() - start
