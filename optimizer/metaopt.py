import tensorflow as tf


def weights_sum(n):
    return tf.ones([n])


def weights_mean(n):
    return tf.ones([n]) / n


@tf.function
def train_meta(
        learner, problem, optimizer, unroll_weights):
    """Meta training on a single problem

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
    """

    problem.reset(tf.size(unroll_weights))

    if problem.dataset is None:
        optimizer.minimize(
            lambda: learner.meta_loss(problem, unroll_weights),
            learner.trainable_variables)
    else:
        for batch in problem.dataset:
            optimizer.minimize(
                lambda: learner.meta_loss(
                    problem, unroll_weights, data=batch),
                learner.trainable_variables)


@tf.function
def train_imitation(
        learner, teacher, student_cpy, teacher_cpy, optimizer, unroll_weights):
    """Imitation learning on a single problem

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
    """

    loss_args = (student_cpy, teacher_cpy, teacher, unroll_weights)

    student_cpy.reset(tf.size(unroll_weights), copy=teacher_cpy)

    if student_cpy.dataset is None:
        optimizer.minimize(
            lambda: learner.imitation_loss(*loss_args),
            learner.trainable_variables)
    else:
        for batch in student_cpy.dataset:
            optimizer.minimize(
                lambda: learner.imitation_loss(*loss_args, data=batch),
                learner.trainable_variables)


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
    """

    for itr, problem in enumerate(problems):

        built = problem.build()
        problem.print(itr)

        # Everything beyond here is a @tf.function
        if teacher is None:
            for _ in range(repeat):
                train_meta(
                    learner, built, optimizer,
                    unroll_weights(unroll()))
        else:
            copy = built.clone_problem()
            for _ in range(repeat):
                train_imitation(
                    learner, teacher, built, copy, optimizer,
                    unroll_weights(unroll()))
