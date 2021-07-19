"""Generic module deserialization."""


def generic(
        target, namespace, pass_cond=None, message="deserialization object",
        default=None):
    """Deserialize object similarly to keras .get.

    Parameters
    ----------
    target : object
        target deserialization object.
    namespace : object
        namespace to search in.

    Keyword Args
    ------------
    pass_cond : callable(object) -> bool or None
        Returns ``True`` if ``target`` is a valid passthrough object. If None,
        never pass through.
    message : str
        Deserialization type for error message.
    default : object
        Default return value (if ``target`` is None).

    Returns
    -------
    object
        Depends on type(target).
        pass_cond(target) == True -> ``target`` passes through.
        None -> returns ``default``.
        str -> fetches object with name ``target`` from namespace.
    """
    if pass_cond and pass_cond(target):
        return target
    if target is None:
        return default
    try:
        return getattr(namespace, target)
    except AttributeError:
        raise ValueError("Invalid {}: {}".format(message, target))
