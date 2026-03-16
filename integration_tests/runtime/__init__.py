"""Module to be imported from at runtime, ie, the callables definitions."""


def check_numpy_version(expected: str) -> bool:
    import numpy

    if numpy.__version__ != expected:
        raise ValueError(f"version check failure, {numpy.__version__=} != {expected=}")
    return True
