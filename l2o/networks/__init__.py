"""Learn to Optimize Networks

Todo: insert refs
"""

from .deepmind_2016 import DMOptimizer
from .scale_basic_2017 import ScaleBasicOptimizer
from .rnnprop_2016 import RNNPropOptimizer
from .scale_hierarchical_2017 import ScaleHierarchicalOptimizer

__all__ = [
    "DMOptimizer",
    "RNNPropOptimizer",
    "ScaleBasicOptimizer",
    "ScaleHierarchicalOptimizer"
]
