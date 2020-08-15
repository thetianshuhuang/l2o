"""Custom command line parsing utility."""

import sys


class ArgParser:
    """Simple custom command line parser."""

    def __init__(self, argv):

        self.args = []
        self.kwargs = {}

        for arg in argv:
            if arg.startswith("--"):
                key, value = arg.split('=') if '=' in arg else (arg, None)
                self.kwargs[key] = value
            else:
                self.args.append(arg)

    def __eval_or_str(self, x):
        """Evaluate or cast to string."""
        try:
            return eval(x)
        except Exception:
            return x

    def pop_check(self, arg):
        """Check if arg is in kwargs, and remove from args if it is."""
        if arg in self.kwargs:
            del self.kwargs[arg]
            return True
        else:
            return False

    def pop_get(self, arg, default=None):
        """Fetch arg from kwargs, and remove from kwargs if present."""
        if arg in self.kwargs:
            res = self.kwargs[arg]
            del self.kwargs[arg]
            return res
        else:
            return default

    def to_overrides(self):
        """Convert argument dictionary to list of overrides."""
        return [
            (path[2:].split('/'), self.__eval_or_str(value))
            for path, value in self.kwargs.items()]
