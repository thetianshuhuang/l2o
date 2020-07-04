from optimizer import DMOptimizer, CoordinateWiseOptimizer, train
from problems import Quadratic, ProblemSpec
import tensorflow as tf


problems = [ProblemSpec(Quadratic, [20], {})]
net = DMOptimizer()
opt = CoordinateWiseOptimizer(net)

train(
    opt, problems, tf.keras.optimizers.Adam(), repeat=10)


test = Quadratic(20)
print(test.objective(None))
for _ in range(20):
    opt.minimize(test.objective(None))
print(test.objective(None))
