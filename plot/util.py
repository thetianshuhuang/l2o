"""Plotting utilities."""

from types import SimpleNamespace
import os
import json


with open('results/names.json') as f:
    FULL_NAMES = json.load(f)


RESULTS = os.listdir('results')
BASELINES = os.listdir('baseline')


def get_test(preset, data='evaluation', period=99):
    """Get test path."""
    if preset in RESULTS:
        if data == 'evaluation':
            if 'repeat' in preset:
                return "results/{}/period_{}.0/conv_train.npz".format(preset, period)
            else:
                return "results/{}/period_{}/conv_train.npz".format(preset, period)
        elif data == 'summary':
            return "results/{}/summary.csv".format(preset)
    elif preset in BASELINES:
        return "baseline/{}/conv_train.npz".format(preset)
    else:
        print(RESULTS)
        print(preset)
        raise KeyError("Invalid test.")


def get_name(test):
    """Get test display name."""
    try:
        return FULL_NAMES[test]
    except KeyError:
        return test
    return test
