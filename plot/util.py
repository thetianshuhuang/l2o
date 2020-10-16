"""Plotting utilities."""

from types import SimpleNamespace
import os
import json


with open('results/names.json') as f:
    FULL_NAMES = json.load(f)


RESULTS = os.listdir('results')


def get_test(test, data='evaluation'):
    """Get test path."""
    if test in RESULTS:
        if data == 'evaluation':
            return "results/{}/period_19.npz".format(test)
        elif data == 'summary':
            return "results/{}/summary.csv".format(test)
    if 'baseline_{}.npz'.format(test) in RESULTS:
        return "results/baseline_{}.npz".format(test)
    else:
        print(RESULTS)
        print(test)
        raise KeyError("Invalid test.")


def get_name(test):
    """Get test display name."""
    try:
        return FULL_NAMES[test]
    except KeyError:
        return test
