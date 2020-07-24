
import sys

from . import strategies
from .. import optimizer


def override(config, path, value):
    """Helper function to programmatically set values in dict"""

    config_ = config
    try:
        for key in path[:-1]:
            if type(config_) == dict:
                config_ = config_[key]
            elif type(config_) == list or type(config_) == tuple:
                config_ = config_[int(key)]
            else:
                raise TypeError(
                    "Config is not a list or dict: {}".format(config_))
    except KeyError:
        raise Exception(
            "Path {} does not exist in object:\n{}".format(
                "/".join(path), str(config)))
    config_[path[-1]] = value


def build(config, overrides):
    """Build learner and learning strategy

    Parameters
    ----------
    config : dict
        Configuration dictionary
    overrides : list
        Override list of (path: str[], value) to pass to ``override``.

    Returns
    -------
    strategy.BaseStrategy
        Initialized strategy with a ``train`` method.
    """

    # Process overrides
    for path, value in overrides:
        override(config, path, value)

    # Initialize network
    if type(config["constructor"]) == str:
        try:
            nn = config["constructor"] + "Optimizer"
            network = getattr(optimizer, nn)
        except AttributeError:
            raise ValueError("L2O algorithm does not exist: {}".format(nn))
    else:
        network = config["constructor"]
    network = network(**config["network"])

    # Initialize architecture (bound to ``architecture`` attribute)
    learner = network.architecture(network, **config["loss_args"])

    # Initialize strategy
    if type(config["strategy_constructor"] == str):
        try:
            sn = config["strategy_constructor"] + "Strategy"
            strategy = getattr(strategies, sn)
        except AttributeError:
            raise ValueError("Training strategy does not exist: {}".format(sn))
    else:
        strategy = config["strategy_constructor"]
    strategy = strategy(
        learner,
        optimizer=config["optimizer"], train_args=config["training"],
        problems=config["problems"], directory=config["directory"],
        **config["strategy"])

    return strategy


def build_argv(config):
    """Build from command line arguments.

    NOTE: this method uses eval, and MUST not be run in a deployed context.

    Parameters
    ----------
    config : dict
        Default arguments

    Returns
    -------
    strategy.BaseStrategy
        Initialized strategy with a ``train`` method.
    """

    # Path -> split by '/'
    # Value -> evaluate to allow float, int, bool, lambda function.
    overrides = [
        (path.split('/'), eval(value)) for path, value in
        [arg.split('=') for arg in sys.argv[1:]]
    ]

    return build(config, overrides)
