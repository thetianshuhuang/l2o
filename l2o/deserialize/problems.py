"""Deserialize problem set."""

import l2o


def _deserialize_problem(p):
    """Helper function to deserialize a problem into a ProblemSpec."""
    if isinstance(p, l2o.problems.ProblemSpec):
        return p
    else:
        try:
            target = p['target']
            if type(target) == str:
                target = getattr(l2o.problems, target)
            return l2o.problems.ProblemSpec(target, p['args'], p['kwargs'])
        except Exception as e:
            raise TypeError(
                "Problem could not be deserialized: {}\n{}".format(p, e))


def problems(pset, default=None):
    """Helper function to _deserialize_problem over a list."""
    if pset is not None:
        return [_deserialize_problem(p) for p in pset]
    else:
        return default
