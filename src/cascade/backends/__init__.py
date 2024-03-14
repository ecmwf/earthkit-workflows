import functools
import xarray as xr
import warnings

from .xarray import XArrayBackend
from .arrayapi import ArrayAPIBackend

BACKENDS = {
    xr.DataArray: XArrayBackend,
    xr.Dataset: XArrayBackend,
    "default": ArrayAPIBackend,
}


def register(type, backend):
    if type in BACKENDS:
        warnings.warn(
            f"Overwriting backend for {type}. Existing backend {BACKENDS[type]}."
        )
    BACKENDS[type] = backend


def array_module(*arrays):
    array_type = type(arrays[0])
    assert all([array_type == type(arrays[x]) for x in range(1, len(arrays))])
    backend = BACKENDS.get(array_type, None)
    if backend is None:
        # Fall back on array API
        backend = BACKENDS["default"]
    return backend


def __getattr__(name: str) -> callable:
    if not hasattr(Backend, name):

        def f(*args, **kwargs):
            backend = array_module(*args)
            return getattr(backend, name)(*args, **kwargs)

        f.__name__ = name
        setattr(Backend, name, f)

    return getattr(Backend, name)


##############################################################################
# Internals


def num_args(expect: int, accept_nested: bool = True):
    """
    Decorator to check the number of arguments passed to a function.
    If expect is -1, then an unlimited number of arguments is allowed.

    Params
    ------
    expect: int, number of arguments expected
    accept_nested: bool, whether to unpack
    """

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


def batchable(func: callable) -> callable:
    """
    Decorator to mark a function as batchable. A method is batchable if
    it can be computed sequentially trivially by applying the same function
    to each batch and to aggregate the batches. Examples of batchable
    functions are sum, prod, min and non-batchable are mean and std.
    """
    func.batchable = True
    return func


class Backend:
    def trivial(arg):
        return arg

    def mean(*args, **kwargs):
        return array_module(*args).mean(*args, **kwargs)

    def std(*args, **kwargs):
        return array_module(*args).std(*args, **kwargs)

    @batchable
    def max(*args, **kwargs):
        return array_module(*args).max(*args, **kwargs)

    @batchable
    def min(*args, **kwargs):
        return array_module(*args).min(*args, **kwargs)

    @batchable
    def sum(*args, **kwargs):
        return array_module(*args).sum(*args, **kwargs)

    @batchable
    def prod(*args, **kwargs):
        return array_module(*args).prod(*args, **kwargs)

    @batchable
    def var(*args, **kwargs):
        return array_module(*args).var(*args, **kwargs)

    def stack(*args, axis: int | None = None, **kwargs):
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
        return array_module(*args).stack(*args, axis=axis, **kwargs)

    @batchable
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
        return array_module(*args).concat(*args, **kwargs)

    @num_args(2)
    def add(*args, **kwargs):
        return array_module(args[0]).add(*args, **kwargs)

    @num_args(2)
    def subtract(*args, **kwargs):
        return array_module(args[0]).subtract(*args, **kwargs)

    @num_args(2)
    def multiply(*args, **kwargs):
        return array_module(*args).multiply(*args, **kwargs)

    @num_args(2)
    def divide(*args, **kwargs):
        return array_module(args[0]).divide(*args, **kwargs)

    @num_args(2)
    def pow(*args, **kwargs):
        return array_module(args[0]).pow(*args, **kwargs)

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
        return array_module(array).take(array, indices, axis=axis, **kwargs)
