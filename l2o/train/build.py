
import os
import sys

import pprint
import json

from . import strategies
from .. import networks


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


def __deep_warn_equal(path, d1, d2, d1name, d2name):
    """Print warning if two structures are not equal (deeply)"""

    if type(d1) == dict:
        iterator = d1
    else:
        if len(d1) != len(d2):
            return ["Warning: <{}> has length {} in {} but {} in {}".format(
                path, len(d1), d1name, len(d2), d2name)]
        iterator = range(len(d1))

    warnings = []
    for key in iterator:
        inner_path = path + "/" + str(key)
        if key not in d2:
            warnings.append(
                "Warning: <{}> is present in {} but not in {}".format(
                    inner_path, d1name, d2name))
        elif not isinstance(d1[key], type(d2[key])):
            warnings.append(
                "Warning: <{}> has type {} in {} but {} in {}".format(
                    inner_path, type(d1[key]), d1name, type(d2[key]), d2name))
        elif type(d1[key]) in (list, tuple, dict):
            warnings += __deep_warn_equal(
                inner_path, d1[key], d2[key], d1name, d2name)
        elif d1[key] != d2[key]:
            warnings.append(
                "Warning: <{}> has value '{}' in {} but '{}'' in {}".format(
                    inner_path, d1[key], d1name, d2[key], d2name))
    return warnings


def deep_warn_equal(d1, d2, d1name, d2name, strict=False):
    warnings = __deep_warn_equal("config", d1, d2, d1name, d2name)
    if len(warnings) > 0:
        wstring = (
            "Specified configuration does not match saved configuration "
            "{}:\n{}\n".format(d2name, '\n'.join(warnings)))
        if strict:
            raise ValueError(wstring)
        else:
            print(wstring)


def build(config, overrides, strict=False):
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

    # Show & save config
    print("Configuration:")
    pprint.pprint(config)
    with open(os.path.join(config["directory"], "config.json"), 'w') as f:
        json.dump(config, f, indent=4)
    print("saved to <{}/config.json>.".format(config["directory"]))
    print("\n")

    # Check saved config
    saved_config = os.path.join(config["directory"], "config.json")
    if os.path.exists(saved_config):
        with open(saved_config) as f:
            config_old = json.load(f)
        deep_warn_equal(
            config, config_old, "config", saved_config, strict=strict)

    # Initialize network
    if type(config["constructor"]) == str:
        try:
            nn = config["constructor"] + "Optimizer"
            network = getattr(networks, nn)
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

    def eval_or_str(x):
        try:
            return eval(x)
        except Exception:
            return x

    # Path -> split by '/'
    # Value -> evaluate to allow float, int, bool, lambda function.
    overrides = [
        (path.split('/'), eval_or_str(value)) for path, value in
        [arg.split('=') for arg in sys.argv[1:]]
    ]

    return build(config, overrides)
