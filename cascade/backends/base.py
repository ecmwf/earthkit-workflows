import functools


def num_args(expect: int, accept_nested: bool = True):
    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def check_num_args(*args, **kwargs):
            if accept_nested and len(args) == 1:
                args = args[0]
            assert (
                len(args) == expect
            ), f"{func.__name__} expects two input arguments, got {len(args)}"
            return func(*args, **kwargs)

        return check_num_args

    return decorator


class BaseBackend:
    def trivial(arg):
        return arg

    def mean(*args, **kwargs):
        raise NotImplementedError()

    def std(*args, **kwargs):
        raise NotImplementedError()

    def maximum(*args, **kwargs):
        raise NotImplementedError()

    def minimum(*args, **kwargs):
        raise NotImplementedError()

    def sum(*args, **kwargs):
        raise NotImplementedError()

    def product(*args, **kwargs):
        raise NotImplementedError()

    def variance(*args, **kwargs):
        raise NotImplementedError()

    def stack(*args, axis: int, **kwargs):
        raise NotImplementedError()

    def concatenate(*args, **kwargs):
        raise NotImplementedError()

    @num_args(2)
    def add(*args, **kwargs):
        return args[0] + args[1]

    @num_args(2)
    def subtract(*args, **kwargs):
        return args[0] - args[1]

    @num_args(2)
    def diff(*args, **kwargs):
        return args[1] - args[0]

    @num_args(2)
    def multiply(*args, **kwargs):
        return args[0] * args[1]

    @num_args(2)
    def divide(*args, **kwargs):
        return args[0] / args[1]

    def take(array, indices, *, axis: int, **kwargs):
        raise NotImplementedError()
