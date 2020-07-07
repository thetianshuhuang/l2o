import l2o
import tensorflow as tf


quad = [l2o.problems.ProblemSpec(
    l2o.problems.Quadratic, [20], {}
)]
mlp = [l2o.problems.ProblemSpec(
    l2o.problems.mlp_classifier, [],
    {"layers": [128, ], "dataset": "mnist", "activation": "relu"}
)]


def create():

    net = l2o.networks.DMOptimizer()
    opt = l2o.optimizer.CoordinateWiseOptimizer(net)

    opt.save("dmoptimizer")


def load():
    return l2o.optimizer.CoordinateWiseOptimizer(
        l2o.networks.DMOptimizer(), weights_file="dmoptimizer")


def train(problems, repeat=1000):

    opt = load()
    l2o.optimizer.train(
        opt, problems, tf.keras.optimizers.Adam(), repeat=repeat)
    opt.save("dmoptimizer")


def test_quadratic(opt):
    problem = l2o.problems.Quadratic(20)
    start = problem.objective(None)
    for _ in range(100):
        opt.minimize(
            lambda: problem.objective(None), problem.trainable_variables)
    print("{} -> {}".format(start, problem.objective(None)))


def test_classify(opt):
    problem = l2o.problems.mlp_classifier(
        layers=[128, ], dataset="kmnist", activation="relu")
    problem.model.compile(
        opt,
        tf.keras.losses.SparseCategoricalCrossentropy())
    problem.model.fit(problem.dataset.batch(32), epochs=5)

    problem.model.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.SparseCategoricalCrossentropy())
    problem.model.fit(problem.dataset.batch(32), epochs=5)
