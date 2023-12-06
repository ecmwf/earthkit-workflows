import array_api_compat
import functools


def trivial(arg):
    return arg


def multi_arg_function(name: str, *args, **kwargs):
    xp = array_api_compat.array_namespace(*args)
    return getattr(xp, name)(args, **kwargs)


def two_arg_function(name: str, *args, **kwargs):
    xp = array_api_compat.array_namespace(*args)
    return getattr(xp, name)(*args, **kwargs)


def method(name: str, array, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)


mean = functools.partial(multi_arg_function, "mean")
std = functools.partial(multi_arg_function, "std")
maximum = functools.partial(multi_arg_function, "max")
minimum = functools.partial(multi_arg_function, "min")
stack = functools.partial(multi_arg_function, "stack")
concatenate = functools.partial(multi_arg_function, "concatenate")
subtract = functools.partial(two_arg_function, "subtract")
add = functools.partial(two_arg_function, "add")
multiply = functools.partial(two_arg_function, "multiply")
divide = functools.partial(two_arg_function, "divide")
__getitem__ = functools.partial(method, "__getitem__")
