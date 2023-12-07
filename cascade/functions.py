import array_api_compat
import functools


def trivial(arg):
    return arg


def multi_arg_function(name: str, *args, **kwargs):
    print(name, len(args), args[0].shape)
    xp = array_api_compat.array_namespace(*args)
    print("XP", xp)
    if len(args) > 1:
        kwargs["axis"] = 0
    return getattr(xp, name)(args, **kwargs)


def two_arg_function(name: str, *args, **kwargs):
    xp = array_api_compat.array_namespace(*args)
    if len(args) == 1:
        args = args[0]
    assert len(args) == 2
    return getattr(xp, name)(args[0], args[1], **kwargs)


def method(name: str, array, *args, **kwargs):
    return getattr(array, name)(*args, **kwargs)


mean = functools.partial(multi_arg_function, "mean")
std = functools.partial(multi_arg_function, "std")
maximum = functools.partial(multi_arg_function, "max")
minimum = functools.partial(multi_arg_function, "min")
sum = functools.partial(multi_arg_function, "sum")
product = functools.partial(multi_arg_function, "prod")
variance = functools.partial(multi_arg_function, "var")
stack = functools.partial(multi_arg_function, "stack")
concatenate = functools.partial(multi_arg_function, "concatenate")

subtract = functools.partial(two_arg_function, "subtract")
add = functools.partial(two_arg_function, "add")
multiply = functools.partial(two_arg_function, "multiply")
divide = functools.partial(two_arg_function, "divide")

__getitem__ = functools.partial(method, "__getitem__")
take = functools.partial(method, "take")
