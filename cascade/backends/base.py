import functools
from abc import ABC, abstractmethod


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


class BaseBackend(ABC):
    def trivial(arg):
        return arg

    @abstractmethod
    def mean(*args, **kwargs):
        pass

    @abstractmethod
    def std(*args, **kwargs):
        pass

    @abstractmethod
    def max(*args, **kwargs):
        pass

    @abstractmethod
    def min(*args, **kwargs):
        pass

    @abstractmethod
    def sum(*args, **kwargs):
        pass

    @abstractmethod
    def prod(*args, **kwargs):
        pass

    @abstractmethod
    def var(*args, **kwargs):
        pass

    @abstractmethod
    def stack(*args, axis: int, **kwargs):
        """
        Join arrays along new axis. All arrays must have
        the same shape, or be broadcastable to the same shape.

        Parameters
        ----------
        arrays: list of Arrays to apply function on
        axis: int | None, axis of new dimension if provided
        method_kwargs: dict, kwargs for array module stack method

        Return
        ------
        Array
        """

    @abstractmethod
    def concat(*args, **kwargs):
        """
        Join along existing axis in one of the inputs

        Parameters
        ----------
        arrays: list of Arrays to apply function on
        method_kwargs: dict, kwargs for array module concatenate method

        Return
        ------
        Array
        """

    @num_args(2)
    def add(*args, **kwargs):
        return args[0] + args[1]

    @num_args(2)
    def subtract(*args, **kwargs):
        return args[0] - args[1]

    @num_args(2)
    def multiply(*args, **kwargs):
        return args[0] * args[1]

    @num_args(2)
    def divide(*args, **kwargs):
        return args[0] / args[1]

    @abstractmethod
    def take(array, indices, *, axis: int, **kwargs):
        """
        Take elements from array specified by indices along
        the specified axis. If indices is an integer, then an array
        with one less dimension is return. If indices is an array
        then the shape of the output matches the input, except along
        axis where it will have the same length as indices

        Parameters
        ----------
        array: Array to take elements from
        indices: int or Array of int, elements to extract from array
        axis: int, axis along which to take elements

        Return
        ------
        Array
        """
