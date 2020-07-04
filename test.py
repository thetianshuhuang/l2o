from optimizer import DMOptimizer, CoordinateWiseOptimizer, train
from problems import Quadratic, ProblemSpec
import tensorflow as tf


problems = [ProblemSpec(Quadratic, [20], {})]
net = DMOptimizer()
opt = CoordinateWiseOptimizer(net)

train(
    opt, problems, tf.keras.optimizers.Adam(), repeat=1000)


def test(log=False):
    test = Quadratic(20)
    start = test.objective(None)
    for _ in range(100):
        opt.minimize(lambda: test.objective(None), test.trainable_variables)
        if log:
            print(test.objective(None))
    print("{} -> {}".format(start, test.objective(None)))
