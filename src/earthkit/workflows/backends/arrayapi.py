# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional, Tuple

from .base import Backend


def _xp_multi_args(name: str, *arrays, axis: int | Tuple[int, ...] | None = None, keepdims: bool = False):
    import array_api_compat

    xp = array_api_compat.array_namespace(*arrays)
    if len(arrays) > 1 and axis is None:
        axis = 0
    else:
        arrays = arrays[0]
    return getattr(xp, name)(xp.asarray(arrays), axis=axis, keepdims=keepdims)


class ArrayAPIBackend(Backend):
    @staticmethod
    def mean(*arrays, backend_kwargs: Optional[dict] = None):
        return _xp_multi_args("mean", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def std(*arrays, backend_kwargs: Optional[dict] = None):
        return _xp_multi_args("std", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def max(*arrays, backend_kwargs: Optional[dict] = None):
        return _xp_multi_args("max", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def min(*arrays, backend_kwargs: Optional[dict] = None):
        return _xp_multi_args("min", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def sum(*arrays, backend_kwargs: Optional[dict] = None):
        return _xp_multi_args("sum", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def prod(*arrays, backend_kwargs: Optional[dict] = None):
        return _xp_multi_args("prod", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def var(*arrays, backend_kwargs: Optional[dict] = None):
        return _xp_multi_args("var", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def stack(*arrays, backend_kwargs: Optional[dict] = None):
        import array_api_compat

        xp = array_api_compat.array_namespace(*arrays)
        broadcasted = xp.broadcast_arrays(*arrays)
        backend_kwargs = backend_kwargs or {}
        backend_kwargs.setdefault("axis", 0)
        return xp.stack(broadcasted, **backend_kwargs)

    @staticmethod
    def concat(*arrays, backend_kwargs: Optional[dict] = None):
        import array_api_compat

        xp = array_api_compat.array_namespace(*arrays)
        return xp.concat(arrays, **(backend_kwargs or {}))

    @staticmethod
    def add(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.add does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 + arr2

    @staticmethod
    def subtract(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.subtract does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 - arr2

    @staticmethod
    def multiply(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.multiply does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 * arr2

    @staticmethod
    def divide(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.divide does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 / arr2

    @staticmethod
    def pow(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.pow does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1**arr2

    @staticmethod
    def take(array, indices, dim: Optional[str | int] = None, *, backend_kwargs: Optional[dict] = None):
        import array_api_compat

        if not isinstance(dim, int):
            raise ValueError("Must provide `dim` as an integer")
        xp = array_api_compat.array_namespace(array)

        if hasattr(indices, "__iter__"):
            return xp.take(array, indices, axis=dim, **(backend_kwargs or {}))
        ret = xp.take(array, [indices], axis=dim, **(backend_kwargs or {}))
        return xp.squeeze(ret, axis=dim)
