"""Learned optimizer packaging."""

import os
import json
from shutil import copyfile

from l2o.strategy import build_from_config


def package(src, dst):
    """Package an L2O for distribution.

    Creates a ``config.json`` file, and copies {}.index and {}.data* checkpoint
    files to weights.index and weights.data*.

    Parameters
    ----------
    src : str
        Source directory. Should be a single replicate.
    dst : str
        Destination directory. If dst already exists, raises FileExistsError.
    """
    # Assemble strategy
    strategy = build_from_config(src)
    checkpoint_info = strategy._complete_metadata({})
    export_chk_base = strategy._path(dtype="checkpoint", **checkpoint_info)

    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg["metadata"] = checkpoint_info

    # Assemble warmup config
    cfg["settings"] = {
        "warmup": strategy.validation_warmup,
        "warmup_rate": strategy.validation_warmup_rate
    }

    # Create directory
    os.makedirs(dst)
    for sfx in [".index", ".data-00000-of-00001"]:
        copyfile(export_chk_base + sfx, os.path.join(dst, "weights" + sfx))

    # Save config
    with open(os.path.join(dst, "config.json"), 'w') as f:
        json.dump(cfg, f, indent=4)
