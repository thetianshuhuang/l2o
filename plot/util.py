"""Plotting utilities."""

from types import SimpleNamespace
import os
import json


with open('results/names.json') as f:
    FULL_NAMES = json.load(f)


RESULTS = os.listdir('results')
BASELINES = os.listdir('baseline')


def get_test(preset, data='evaluation'):
    """Get test path."""
    if ':' in preset:
        preset, period = preset.split(':')
    else:
        period = 99
    if preset in RESULTS:
        if data == 'evaluation':
            p = "results/{}/period_{}/conv_train.npz".format(preset, period)
            if os.path.exists(p):
                return p
            else:
                return "results/{}/period_{}.0/conv_train.npz".format(preset, period)
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
        if ':' in test:
            base, period = test.split(':')
            return FULL_NAMES[base] + " @ {}".format(period)
        else:
            return FULL_NAMES[test]
    except KeyError:
        if ':' in test:
            base, period = test.split(':')
            return base + "@{}".format(period)
        else:
            return test
