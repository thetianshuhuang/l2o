"""Build from saved config."""

import os
import json
import pprint

import l2o
from l2o.train import OptimizerTraining


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


def build(config, overrides, directory="weights", strict=True, info=True):
    """Build learner, training, and strategy.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    overrides : (path, value)[]
        Override list to pass to ``override``.

    Keyword Args
    ------------
    directory : str
        Directory to run inside / save to.
    strict : bool
        If True, raises exception if config.json is already present and does
        not match ``config``.
    info : bool
        Flag to disable printing out config. Warnings/errors are not affected.

    Returns
    -------
    strategy.Strategy
        Strategy built according to ``config`` and ``overrides``.
    """
    for path, value in overrides:
        override(config, path, value)

    # Check saved config or save config
    saved_config = os.path.join(directory, "config.json")
    if os.path.exists(saved_config):
        with open(saved_config) as f:
            config_old = json.load(f)
        deep_warn_equal(
            config, config_old, "config", saved_config, strict=strict)
    else:
        os.makedirs(directory, exist_ok=True)
        with open(saved_config, 'w') as f:
            json.dump(config, f, indent=4)
        print("Config saved to <{}/config.json>.".format(directory))

    if info:
        print("Configuration:")
        pprint.pprint(config)

    # Build optimizer policy
    policy_constructor = l2o.deserialize.generic(
        config["policy_constructor"], l2o.policies, pass_cond=None,
        message="learned optimizer model", default=l2o.policies.DMOptimizer)
    policy = policy_constructor(**config["policy"])

    # Build learner
    learner = OptimizerTraining(
        policy, config["optimizer"], **config["training"])

    # Build strategy
    strategy_constructor = l2o.deserialize.generic(
        config["strategy_constructor"], l2o.strategy, pass_cond=None,
        message="meta learning strategy", default=l2o.strategy.SimpleStrategy)
    strategy = strategy_constructor(
        learner, config["problems"], directory=directory, **config["strategy"])

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

    return build(config, [], directory=directory, strict=False)
