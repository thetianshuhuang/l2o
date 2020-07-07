from l2o.networks import DMOptimizer
from l2o.optimizer import CoordinateWiseOptimizer, train
from l2o.problems import Quadratic, ProblemSpec
import tensorflow as tf


def create():

    problems = [ProblemSpec(Quadratic, [20], {})]
    net = DMOptimizer()
    opt = CoordinateWiseOptimizer(net)

    train(
        opt, problems, tf.keras.optimizers.Adam(), repeat=1000)

    opt.save("dmoptimizer")


def load():
    return CoordinateWiseOptimizer(DMOptimizer(), weights_file="dmoptimizer")


def test(opt, log=False):
    test = Quadratic(20)
    start = test.objective(None)
    for _ in range(100):
        opt.minimize(lambda: test.objective(None), test.trainable_variables)
        if log:
            print(test.objective(None))
    print("{} -> {}".format(start, test.objective(None)))
