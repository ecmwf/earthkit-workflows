import array_api_compat
import functools

from .common import *


def multi_arg_function(name: str, *arrays, **method_kwargs):
    """
    Apply named function in array module on arrays. If arrays
    only consists of single array then function is applied
    along an axis (default is axis 0) of this single array.

    Parameters
    ----------
    name: str, name of function to apply
    arrays: ArrayLike, list of arrays to apply function on

    Return
    ------
    ArrayLike, result of applying named function
    """
    if len(arrays) > 1:
        method_kwargs["axis"] = 0
    else:
        arrays = arrays[0]
        # method_kwargs.setdefault("axis", 0)

    xp = array_api_compat.array_namespace(*arrays)
    return getattr(xp, name)(arrays, **method_kwargs)


def two_arg_function(name: str, *arrays, **method_kwargs):
    """
    Apply named function in array module on arrays. If arrays
    only consists of single array then function is applied
    along an axis 0 of this single array.

    Parameters
    ----------
    name: str, name of function to apply
    arrays: ArrayLike, list of arrays to apply function on

    Return
    ------
    ArrayLike, result of applying named function

    Raises
    ------
    AssertionError if more than two arrays are passed as inputs, or if axis 0
    of the single array argument does not have dimension 2
    """
    if len(arrays) == 1:
        arrays = arrays[0]
    assert (
        len(arrays) == 2
    ), f"two_arg_function {name} expects two input arguments, got {len(arrays)}"

    xp = array_api_compat.array_namespace(*arrays)
    return getattr(xp, name)(arrays[0], arrays[1], **method_kwargs)


def single_arg_function(name: str, array, **method_kwargs):
    """
    Apply named function in array module on array

    Parameters
    ----------
    name: str, name of function to apply
    array: ArrayLike

    Return
    ------
    ArrayLike, result of applying named function
    """
    xp = array_api_compat.array_namespace(array)
    return getattr(xp, name)(array, **method_kwargs)


mean = functools.partial(multi_arg_function, "mean")
std = functools.partial(multi_arg_function, "std")
maximum = functools.partial(multi_arg_function, "max")
minimum = functools.partial(multi_arg_function, "min")
sum = functools.partial(multi_arg_function, "sum")
product = functools.partial(multi_arg_function, "prod")
variance = functools.partial(multi_arg_function, "var")
stack = functools.partial(multi_arg_function, "stack")
concatenate = functools.partial(multi_arg_function, "concat")

subtract = functools.partial(two_arg_function, "subtract")
add = functools.partial(two_arg_function, "add")
multiply = functools.partial(two_arg_function, "multiply")
divide = functools.partial(two_arg_function, "divide")

take = functools.partial(single_arg_function, "take")
