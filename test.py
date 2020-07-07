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

    net = l2o.optimizer.DMOptimizer()
    opt = l2o.optimizer.CoordinateWiseOptimizer(net)

    opt.save("dmoptimizer")


def load():
    return l2o.optimizer.CoordinateWiseOptimizer(
        l2o.optimizer.DMOptimizer(), weights_file="dmoptimizer")


def train(problems):

    opt = load()
    train(opt, problems, tf.keras.optimizers.Adam(), repeat=1000)
    opt.save("dmoptimizer")


def test_quadratic(opt):
    problem = l2o.problems.Quadratic(20)
    start = test.objective(None)
    for _ in range(100):
        opt.minimize(
            lambda: problem.objective(None), problem.trainable_variables)
    print("{} -> {}".format(start, problem.objective(None)))


def test_classify(opt):
    problem = l2o.problems.mlp_classifier(
        layers=[128, ], dataset="kmnist", activation="relu")
    problem.network.fit(problem.dataset.batch(32), epochs=5)
