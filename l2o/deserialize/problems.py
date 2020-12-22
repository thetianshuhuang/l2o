"""Deserialize problem set."""

import l2o


def _deserialize_problem(p):
    """Helper function to deserialize a problem dict into a Problem."""
    if isinstance(p, l2o.problems.Problem):
        return p
    else:
        try:
            target = p['target']
            if type(target) == str:
                target = getattr(l2o.problems, target)
            return target(*p['args'], config=p, **p['kwargs'])
        except Exception as e:
            raise TypeError(
                "Problem could not be deserialized: {}\n{}".format(p, e))


def problems(pset, default=None):
    """Helper function to _deserialize_problem over a list."""
    if pset is not None:
        return [_deserialize_problem(p) for p in pset]
    else:
        return default
