import tensorflow as tf
import time


def weights_sum(n):
    return tf.ones([n])


def weights_mean(n):
    return tf.ones([n]) / n


def train_meta(
        learner, problem, optimizer, unroll_weights, progress):
    """Meta training on a single problem (batched)

    Parameters
    ----------
    learner : trainable_optimizer.TrainableOptimizer
        Optimizer to train
    problem : problem.Problem
        Pre-built problem to train meta-epoch on.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer for meta optimization
    unroll_weights : tf.Tensor
        Unroll weights
    progress : tf.keras.utils.ProgBar
        Can't do without a progress bar!
    """

    problem.reset(tf.size(unroll_weights))
    learner._create_slots(problem.trainable_variables)

    for batch in problem.dataset:
        optimizer.minimize(
            lambda: learner.meta_loss(
                problem, unroll_weights, data=batch),
            learner.trainable_variables)
        progress.add(1)


def train_imitation(
        learner, teacher, student_cpy, teacher_cpy, optimizer, unroll_weights,
        progress):
    """Imitation learning on a single problem (batched)

    Parameters
    ----------
    learner : trainable_optimizer.TrainableOptimizer
        Optimizer to train
    teacher : tf.keras.optimizers.Optimizer
        Optimizer to emulate
    student_cpy : problem.Problem
        Student copy of the problem to train on
    teacher_cpy : problem.Problem
        Teacher copy of the problem to train on. Must be an exact copy of the
        student problem.
    optimizer : tf.keras.optimizers.Optimizer
        Optimizer for meta optimization
    unroll_weights : tf.Tensor
        Unroll weights
    progress : tf.keras.utils.ProgBar
        Can't do without a progress bar!
    """

    # Reset problem
    student_cpy.reset(tf.size(unroll_weights), copy=teacher_cpy)

    # Reset optimizers
    learner._create_slots(student_cpy.trainable_variables)
    # Teacher (i.e. Adam) must be manually reset
    # We assume here that ``teacher`` is a tf.keras.optimizers built-in that
    # only overwrites tensorflow-approved methods
    for var in teacher.variables():
        var.assign(tf.zeros_like(var))

    for batch in student_cpy.dataset:
        # Sync student up with teacher first every unroll
        student_cpy.sync(teacher_cpy)
        optimizer.minimize(
            lambda: learner.imitation_loss(
                student_cpy, teacher_cpy, teacher, unroll_weights, data=batch),
            learner.trainable_variables)
        progress.add(1)


def train(
        learner, problems, optimizer,
        unroll=lambda _: 20, unroll_weights=weights_mean, teacher=None,
        repeat=1):
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
    unroll : callable (problem.Problem -> int)
        Returns the number of unroll iterations; called once for each problem.
        Could be random!
    unroll_weights : callable (int -> tf.Tensor[unroll])
        Function that creates a tensor encoding iteration weights for the
        truncated BPTT loss calculation
    teacher : tf.keras.optimizers.Optimizer
        Optimizer to emulate. If None, meta training loss is used. Otherwise,
        imitation learning is used.
    repeat : int
        Number of consecutive meta-iterations to repeat each problem for,
        resetting on each iteration.

    Returns
    -------
    float
        Time used during training; can be ignored.
    """

    start = time.time()

    for itr, problem in enumerate(problems):

        built = problem.build()
        problem.print(itr)
        unroll = unroll(problem)
        size = built.get_size(unroll)

        progress = tf.keras.utils.Progbar(
            repeat * size, unit_name='meta-iteration')

        for _ in range(repeat):
            if teacher is None:
                train_meta(
                    learner, built, optimizer,
                    unroll_weights(unroll), progress)
            else:
                copy = built.clone_problem()
                train_imitation(
                    learner, teacher, built, copy, optimizer,
                    unroll_weights(unroll), progress)

    return time.time() - start
