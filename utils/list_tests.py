"""List experiments."""

import os
import sys


def crawl_experiments(d):

    contents = os.listdir(d)
    if "config.json" in contents:
        if "eval" in contents:
            tests = os.path.listdir(os.path.join(d, "eval"))
        else:
            tests = []
        return [(d, tests)]
    else:
        res = []
        for c in contents:
            res += crawl_experiments(os.path.join(d, c))
        return res


target = sys.argv[1]

print("Experiments:")
for exp, tests in crawl_experiments(target):
    print("{}: {}".format(exp, tests))
