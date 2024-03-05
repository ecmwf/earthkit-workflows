import array_api_compat

from .base import BaseBackend
from .decorators import batchable


class ArrayApiBackend(BaseBackend):
    def stat_funcs(name: str, *arrays, **method_kwargs):
        """
        Apply named function in array module on arrays. If arrays
        only consists of single array then function is applied
        along an axis of this single array, if axis kwarg is specified.

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

        xp = array_api_compat.array_namespace(*arrays)
        return getattr(xp, name)(xp.asarray(arrays), **method_kwargs)

    def mean(*arrays, **method_kwargs):
        return ArrayApiBackend.stat_funcs("mean", *arrays, **method_kwargs)

    def std(*arrays, **method_kwargs):
        return ArrayApiBackend.stat_funcs("std", *arrays, **method_kwargs)

    @batchable
    def min(*arrays, **method_kwargs):
        return ArrayApiBackend.stat_funcs("min", *arrays, **method_kwargs)

    @batchable
    def max(*arrays, **method_kwargs):
        return ArrayApiBackend.stat_funcs("max", *arrays, **method_kwargs)

    @batchable
    def sum(*arrays, **method_kwargs):
        return ArrayApiBackend.stat_funcs("sum", *arrays, **method_kwargs)

    @batchable
    def prod(*arrays, **method_kwargs):
        return ArrayApiBackend.stat_funcs("prod", *arrays, **method_kwargs)

    def var(*arrays, **method_kwargs):
        return ArrayApiBackend.stat_funcs("var", *arrays, **method_kwargs)

    def stack(*arrays, axis: int = 0):
        xp = array_api_compat.array_namespace(*arrays)
        broadcasted = xp.broadcast_arrays(*arrays)
        return xp.stack(broadcasted, axis=axis)

    @batchable
    def concat(*arrays, axis: int = 0, **method_kwargs):
        xp = array_api_compat.array_namespace(*arrays)
        return xp.concat(arrays, axis=axis, **method_kwargs)

    def take(array, indices, *, axis: int, **kwargs):
        xp = array_api_compat.array_namespace(array)

        if hasattr(indices, "__iter__"):
            return xp.take(array, indices, axis=axis, **kwargs)
        ret = xp.take(array, [indices], axis=axis, **kwargs)
        return xp.squeeze(ret, axis=axis)
