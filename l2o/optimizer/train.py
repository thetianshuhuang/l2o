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
    problem : problem.Problem
        Problem to train meta-epoch on.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer to use for meta-optimization
    weights : tf.Tensor([unroll])
        Unroll weights.
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


def _train_inner(itr):
    """Meta training or imitation learning on a single problem for a single
    meta-iteration

    Parameters
    ----------
    itr : MetaIteration
        Iteration to run
    """

    # Prepare learner & teacher
    itr.learner._create_slots(itr.problem.trainable_variables)
    if itr.teacher is not None:
        problem_cpy = itr.problem.clone_problem()
        itr.teacher._create_slots(problem_cpy.trainable_variables)

    # Have one progress bar per epoch unless there are too many epochs
    if itr.epochs > 10:
        progress = tf.keras.utils.Progbar(
            itr.epochs * itr.problem.size(itr.unroll), unit_name='step')

    for i in range(itr.epochs):

        if itr.epochs <= 10:
            print("Epoch {}".format(i + 1))
            progress = tf.keras.utils.Progbar(
                itr.problem.size(itr.unroll), unit_name='step')

        # Reset problem and rebatch
        # (both methods optionally implemented by problem)
        dataset = itr.problem.get_dataset(itr.unroll)
        itr.problem.reset()

        for batch in dataset:
            if batch is not None:
                sub_batches = [
                    tf.stack(tf.split(dim, num_or_size_splits=itr.unroll))
                    for dim in batch]
            else:
                sub_batches = None

            with tf.GradientTape() as tape:
                tape.watch(itr.learner.trainable_variables)
                if itr.teacher is None:
                    loss = itr.learner.meta_loss(
                        itr.problem, itr.weights, tf.constant(itr.unroll),
                        data=sub_batches, noise_stddev=itr.noise_stddev)
                else:
                    loss = itr.learner.imitation_loss(
                        itr.problem, problem_cpy, itr.teacher, itr.weights,
                        tf.constant(itr.unroll), data=sub_batches,
                        noise_stddev=itr.noise_stddev)

            grads = tape.gradient(loss, itr.learner.trainable_variables)
            itr.optimizer.apply_gradients(
                zip(grads, itr.learner.trainable_variables))

            import pdb
            pdb.set_trace()

            progress.add(1, values=[("loss", loss)])


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

        built = problem.build()
        problem.print(itr)
        _train_inner(MetaIteration(
            learner=learner,
            teacher=teacher,
            problem=built,
            optimizer=optimizer,
            weights=unroll_weights(unroll),
            unroll=unroll,
            noise_stddev=noise_stddev,
            epochs=epochs,
            idx=itr))

    return time.time() - start
