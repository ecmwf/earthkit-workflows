import functools
import xarray as xr
import array_api_compat

from .xarray import XArrayBackend

BACKENDS = {
    xr.DataArray: XArrayBackend,
    xr.Dataset: XArrayBackend,
}


def register(type, backend):
    assert type not in BACKENDS
    BACKENDS[type] = backend


def __getattr__(name: str) -> callable:
    if not hasattr(Backend, name):

        def f(*args, **kwargs):
            backend = array_module(*args)
            if backend is None:
                backend = array_api_compat.array_namespace(*args)
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


def _xp_multi_args(name: str, *args, **kwargs):
    xp = array_api_compat.array_namespace(*args)
    if len(args) > 1:
        kwargs["axis"] = 0
    else:
        args = args[0]
    return getattr(xp, name)(xp.asarray(args), **kwargs)


def array_module(*arrays):
    array_type = type(arrays[0])
    assert all([array_type == type(arrays[x]) for x in range(1, len(arrays))])
    return BACKENDS.get(array_type, None)


class Backend:
    def trivial(arg):
        return arg

    def mean(*args, **kwargs):
        module = array_module(*args)
        if module is not None:
            return module.mean(*args, **kwargs)

        # Fall back on array api
        return _xp_multi_args("mean", *args, **kwargs)

    def std(*args, **kwargs):
        module = array_module(*args)
        if module is not None:
            return module.std(*args, **kwargs)

        # Fall back on array api
        return _xp_multi_args("std", *args, **kwargs)

    @batchable
    def max(*args, **kwargs):
        module = array_module(*args)
        if module is not None:
            return module.max(*args, **kwargs)

        # Fall back on array api
        return _xp_multi_args("max", *args, **kwargs)

    @batchable
    def min(*args, **kwargs):
        module = array_module(*args)
        if module is not None:
            return module.min(*args, **kwargs)

        # Fall back on array api
        return _xp_multi_args("min", *args, **kwargs)

    @batchable
    def sum(*args, **kwargs):
        module = array_module(*args)
        if module is not None:
            return module.sum(*args, **kwargs)

        # Fall back on array api
        return _xp_multi_args("sum", *args, **kwargs)

    @batchable
    def prod(*args, **kwargs):
        module = array_module(*args)
        if module is not None:
            return module.prod(*args, **kwargs)

        # Fall back on array api
        return _xp_multi_args("prod", *args, **kwargs)

    @batchable
    def var(*args, **kwargs):
        module = array_module(*args)
        if module is not None:
            return module.var(*args, **kwargs)

        # Fall back on array api
        return _xp_multi_args("var", *args, **kwargs)

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
        module = array_module(*args)
        if module is not None:
            return module.stack(*args, axis=axis, **kwargs)

        xp = array_api_compat.array_namespace(*args)
        broadcasted = xp.broadcast_arrays(*args)
        return xp.stack(broadcasted, axis=axis)

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
        module = array_module(*args)
        if module is not None:
            return module.concat(*args, **kwargs)

        xp = array_api_compat.array_namespace(*args)
        return xp.concat(args, **kwargs)

    @num_args(2)
    def add(*args, **kwargs):
        module = array_module(args[0])
        if module is not None:
            return module.add(*args, **kwargs)
        return args[0] + args[1]

    @num_args(2)
    def subtract(*args, **kwargs):
        module = array_module(args[0])
        if module is not None:
            return module.subtract(*args, **kwargs)
        return args[0] - args[1]

    @num_args(2)
    def multiply(*args, **kwargs):
        module = array_module(args[0])
        if module is not None:
            return module.multiply(*args, **kwargs)
        return args[0] * args[1]

    @num_args(2)
    def divide(*args, **kwargs):
        module = array_module(args[0])
        if module is not None:
            return module.divide(*args, **kwargs)
        return args[0] / args[1]

    @num_args(2)
    def pow(*args, **kwargs):
        module = array_module(args[0])
        if module is not None:
            return module.pow(*args, **kwargs)
        return args[0] ** args[1]

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
        module = array_module(array)
        if module is not None:
            return module.take(array, indices, axis=axis, **kwargs)

        xp = array_api_compat.array_namespace(array)

        if hasattr(indices, "__iter__"):
            return xp.take(array, indices, axis=axis, **kwargs)
        ret = xp.take(array, [indices], axis=axis, **kwargs)
        return xp.squeeze(ret, axis=axis)
