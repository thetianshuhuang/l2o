"""Learn to Optimize Networks.

Attributes
----------
BaseLearnToOptimizePolicy
    Base class that all policies should extend.
AdamOptimizer
    Standard Adam optimizer.
RMSPropOptimizer
    Standard RMSProp optimizer.
SGDOptimizer
    Standard SGD optimizer.
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
RNNPropExtendedOptimizer
    Extended version of RNNProp that includes direct gradient input and
    shortcut connections to each layer.
ChoiceExtendedOptimizer
    Extended version of ChoiceOptimizer using the same modifications as
    RNNPropExtended.
"""

from .architectures import BaseLearnToOptimizePolicy
from .deepmind_2016 import DMOptimizer
from .scale_basic_2017 import ScaleBasicOptimizer
from .dynamic_rate import AdamLROptimizer, RMSPropLROptimizer
from .rnnprop_2016 import RNNPropOptimizer
from .scale_hierarchical_2017 import ScaleHierarchicalOptimizer
from .choice import ChoiceOptimizer
from .abstract_choice import AbstractChoiceOptimizer
from .analytical import (
    AdamOptimizer, RMSPropOptimizer, SGDOptimizer,
    MomentumOptimizer, PowerSignOptimizer, AddSignOptimizer,
    AdaptivePowerSignOptimizer, AdaptiveAddSignOptimizer)
from .rnnprop_ext import RNNPropExtendedOptimizer
from .load import load

__all__ = [
    # Utilities & helpers
    "BaseLearnToOptimizePolicy",
    "load",
    # Hand-crafted optimizers
    "AdamOptimizer",
    "RMSPropOptimizer",
    "SGDOptimizer",
    "MomentumOptimizer",
    # Neural Optimizer Search
    "PowerSignOptimizer",
    "AddSignOptimizer",
    "AdaptivePowerSignOptimizer",
    "AdaptiveAddSignOptimizer",
    # Deepmind, Google Research
    "DMOptimizer",
    "ScaleBasicOptimizer",
    "ScaleHierarchicalOptimizer",
    # RNNProp family
    "RNNPropOptimizer",
    "RNNPropExtendedOptimizer",
    # Resticted Direction
    "AdamLROptimizer",
    "RMSPropLROptimizer",
    # Optimizer Choice
    "ChoiceOptimizer",
]
