# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import warnings
from typing import Callable

import xarray as xr

from .arrayapi import ArrayAPIBackend
from .xarray import XArrayBackend

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
    # Only deduce type from first element to allow for mixed types
    # but this means the first argument needs to specify the correct module
    array_type = type(arrays[0])
    backend = BACKENDS.get(array_type, None)
    if backend is None:
        # Fall back on array API
        backend = BACKENDS["default"]
    return backend


def __getattr__(name: str) -> Callable:
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
    """Decorator to check the number of arguments passed to a function.
    If expect is -1, then an unlimited number of arguments is allowed.

    Params
    ------
    expect: int, number of arguments expected
    accept_nested: bool, whether to unpack
    """

    def decorator(func: Callable) -> Callable:
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


def batchable(func: Callable) -> Callable:
    """Decorator to mark a function as batchable. A method is batchable if
    it can be computed sequentially trivially by applying the same function
    to each batch and to aggregate the batches. Examples of batchable
    functions are sum, prod, min and non-batchable are mean and std.
    """
    func.batchable = True  # type: ignore # monkeypatching...
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

    def stack(*args, axis: int = 0, **kwargs):
        """Join arrays along new axis. All arrays must have
        the same shape, or be broadcastable to the same shape.

        Parameters
        ----------
        arrays: list of Arrays to apply function on
        axis: int, axis of new dimension if provided
        method_kwargs: dict, kwargs for array module stack method

        Return
        ------
        Array
        """
        return array_module(*args).stack(*args, axis=axis, **kwargs)

    @batchable
    def concat(*args, **kwargs):
        """Join along existing axis in one of the inputs

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
        return array_module(*args).add(*args, **kwargs)

    @num_args(2)
    def subtract(*args, **kwargs):
        return array_module(*args).subtract(*args, **kwargs)

    @num_args(2)
    def multiply(*args, **kwargs):
        return array_module(*args).multiply(*args, **kwargs)

    @num_args(2)
    def divide(*args, **kwargs):
        return array_module(*args).divide(*args, **kwargs)

    @num_args(2)
    def pow(*args, **kwargs):
        return array_module(*args).pow(*args, **kwargs)

    def take(array, indices, **kwargs):
        """Take elements from array specified by indices along
        the specified axis. If indices is an integer, then an array
        with one less dimension is return. If indices is an array
        then the shape of the output matches the input, except along
        axis where it will have the same length as indices. Axis can
        be specified using axis kwarg, or with dim if using xarray.

        Parameters
        ----------
        array: Array to take elements from
        indices: int or Array of int, elements to extract from array

        Return
        ------
        Array
        """
        return array_module(array).take(array, indices, **kwargs)


try:
    from earthkit.data import FieldList, SimpleFieldList

    from earthkit.workflows.backends.earthkit import FieldListBackend

    BACKENDS[SimpleFieldList] = FieldListBackend
    BACKENDS[FieldList] = FieldListBackend
except ImportError:
    warnings.warn("earthkit could not be imported, FieldList not supported.")
