"""Packaging script."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import l2o


exports = {
    "choice2": "choice-small",
    "choice6": "choice-large",
    "oscale-gs": "min-max",
    "mean": "mean"
}


for basesrc, basedst in exports.items():
    for repl in range(8):
        l2o.distutils.package(
            os.path.join("results", "rnnprop", basesrc, str(repl + 1)),
            os.path.join("pre-trained", basedst, str(repl + 1)))
