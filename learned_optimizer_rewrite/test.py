from optimizer.deepmind_2016 import DMOptimizer
from optimizer.coordinatewise_rnn import CoordinateWiseOptimizer
from problems.problem import Quadratic
import tensorflow as tf


prob = Quadratic(20)
net = DMOptimizer()
opt = CoordinateWiseOptimizer(net)
loss = opt.meta_loss(prob, tf.ones([20]))
