"""Methods related to building optimizers and training strategy."""

import os
import sys

import pprint
import json

from . import strategies
from .. import networks


def override(config, path, value):
    """Helper function to programmatically set values in a nested structure."""
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
    if path[-1] == '*':
        config_.append(value)
    else:
        config_[path[-1]] = value


def __deep_warn_equal(path, d1, d2, d1name, d2name):
    """Print warning if two structures are not equal (deeply)."""
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
        if type(d1) == dict and key not in d2:
            warnings.append(
                "<{}> is present in {} but not in {}".format(
                    inner_path, d1name, d2name))
        elif not isinstance(d1[key], type(d2[key])):
            warnings.append(
                "<{}> has type {} in {} but {} in {}".format(
                    inner_path, type(d1[key]), d1name, type(d2[key]), d2name))
        elif type(d1[key]) in (list, tuple, dict):
            warnings += __deep_warn_equal(
                inner_path, d1[key], d2[key], d1name, d2name)
        elif d1[key] != d2[key]:
            warnings.append(
                "<{}> has value '{}' in {} but '{}'' in {}".format(
                    inner_path, d1[key], d1name, d2[key], d2name))
    return warnings


def deep_warn_equal(d1, d2, d1name, d2name, strict=False):
    """Warn if two nested structures are not (deeply) equal."""
    warnings = __deep_warn_equal("config", d1, d2, d1name, d2name)
    if len(warnings) > 0:
        wstring = (
            "specified configuration does not match saved configuration "
            "{}:\n{}\n".format(d2name, '\n'.join(warnings)))
        if strict:
            raise ValueError(wstring)
        else:
            print("Warning: " + wstring)


def __check_and_save_config(config, directory, strict=True):
    """Check configuration against saved config in specified directory."""
    # Check saved config
    saved_config = os.path.join(directory, "config.json")
    if os.path.exists(saved_config):
        with open(saved_config) as f:
            config_old = json.load(f)
        deep_warn_equal(
            config, config_old, "config", saved_config, strict=strict)

    # Show & save
    print("Configuration:")
    pprint.pprint(config)
    with open(saved_config, 'w') as f:
        json.dump(config, f, indent=4)
    print("saved to <{}/config.json>.".format(directory))


def build(
        config, overrides, directory="weights",
        saved_config=True, strict=True):
    """Build learner and learning strategy.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    overrides : list
        Override list of (path: str[], value) to pass to ``override``.

    Keyword Args
    ------------
    directory : str
        Directory to run inside.
    saved_config : bool
        Check against saved configuration and save configuration to folder
    strict : bool
        If True, enforces strict equality between saved configuration and
        specified configuration on resumed training.

    Returns
    -------
    strategy.BaseStrategy
        Initialized strategy with a ``train`` method.
    """
    for path, value in overrides:
        override(config, path, value)

    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Check, show & save config
    if saved_config:
        __check_and_save_config(config, directory, strict=strict)

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
        problems=config["problems"],
        validation_problems=config.get("validation_problems"),
        directory=directory,
        **config["strategy"])

    return strategy


def build_from_config(directory):
    """Build from saved configuration.

    Parameters
    ----------
    directory : str
        Directory containing saved configuration and data.
    """
    with open(os.path.join(directory, "config.json")) as x:
        config = json.load(x)

    return build(
        config, [], directory=directory, saved_config=False, strict=False)
