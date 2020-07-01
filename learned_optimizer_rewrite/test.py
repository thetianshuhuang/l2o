from .optimizer.deepmind_2016 import DMOptimizer
from .optimizer.coordinatewise_rnn import CoordinateWiseOptimizer


net = DMOptimizer()
opt = CoordinateWiseOptimizer(net)



