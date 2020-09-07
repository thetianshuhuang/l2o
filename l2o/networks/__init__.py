"""Learn to Optimize Networks

Attributes
----------
DMOptimizer
    "Learing to learn by gradient descent by gradient descent"
    (Andrychowicz et. al, 2016)
RNNPropOptimizer
    "Learning Gradient Descent: Better Generalization and Longer Horizons"
    (Lv. et. al, 2017)
ScaleBasicOptimizer
    "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)
ScaleHierarchicalOptimizer
    "Learned Optimizers that Scale and Generalize" (Wichrowska et. al, 2017)
ChoiceOptimizer
    Optimizer that chooses either Adam or RMSProp in a coordinatewise fashion.
"""

from .deepmind_2016 import DMOptimizer
from .scale_basic_2017 import ScaleBasicOptimizer
from .rnnprop_2016 import RNNPropOptimizer
from .scale_hierarchical_2017 import ScaleHierarchicalOptimizer
from .choice import ChoiceOptimizer

__all__ = [
    "DMOptimizer",
    "RNNPropOptimizer",
    "ScaleBasicOptimizer",
    "ScaleHierarchicalOptimizer",
    "ChoiceOptimizer"
]
