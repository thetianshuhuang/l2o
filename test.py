from optimizer import DMOptimizer, CoordinateWiseOptimizer, train
from problems import Quadratic, ProblemSpec
import tensorflow as tf


problems = [ProblemSpec(Quadratic, [20], {})]
net = DMOptimizer()
opt = CoordinateWiseOptimizer(net)

train(
    opt, problems, tf.keras.optimizers.Adam())
