# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import TYPE_CHECKING, Optional, Tuple, TypeAlias

from .base import Backend

if TYPE_CHECKING:
    # TODO: use array_api_typing.Array instead of np.ndarray when available
    import numpy as np

    Array: TypeAlias = "np.ndarray"


def _xp_multi_args(name: str, *arrays: "Array", axis: int | Tuple[int, ...] | None = None, keepdims: bool = False) -> "Array":
    import array_api_compat

    xp = array_api_compat.array_namespace(*arrays)
    array: "Array"
    if len(arrays) > 1 and axis is None:
        axis = 0
        array = xp.asarray(arrays)
    else:
        array = arrays[0]
    return getattr(xp, name)(array, axis=axis, keepdims=keepdims)


class ArrayAPIBackend(Backend):
    @classmethod
    def mean(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        return _xp_multi_args("mean", *arrays, **(backend_kwargs or {}))

    @classmethod
    def std(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        return _xp_multi_args("std", *arrays, **(backend_kwargs or {}))

    @classmethod
    def max(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        return _xp_multi_args("max", *arrays, **(backend_kwargs or {}))

    @classmethod
    def min(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        return _xp_multi_args("min", *arrays, **(backend_kwargs or {}))

    @classmethod
    def sum(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        return _xp_multi_args("sum", *arrays, **(backend_kwargs or {}))

    @classmethod
    def prod(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        return _xp_multi_args("prod", *arrays, **(backend_kwargs or {}))

    @classmethod
    def var(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        return _xp_multi_args("var", *arrays, **(backend_kwargs or {}))

    @classmethod
    def stack(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        import array_api_compat

        xp = array_api_compat.array_namespace(*arrays)
        broadcasted = xp.broadcast_arrays(*arrays)
        backend_kwargs = backend_kwargs or {}
        backend_kwargs.setdefault("axis", 0)
        return xp.stack(broadcasted, **backend_kwargs)

    @classmethod
    def concat(cls, *arrays: "Array", backend_kwargs: Optional[dict] = None) -> "Array":
        import array_api_compat

        xp = array_api_compat.array_namespace(*arrays)
        return xp.concat(arrays, **(backend_kwargs or {}))

    @classmethod
    def add(cls, arr1: "Array", arr2: "Array", *, backend_kwargs: Optional[dict] = None) -> "Array":
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.add does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 + arr2

    @classmethod
    def subtract(cls, arr1: "Array", arr2: "Array", *, backend_kwargs: Optional[dict] = None) -> "Array":
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.subtract does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 - arr2

    @classmethod
    def multiply(cls, arr1: "Array", arr2: "Array", *, backend_kwargs: Optional[dict] = None) -> "Array":
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.multiply does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 * arr2

    @classmethod
    def divide(cls, arr1: "Array", arr2: "Array", *, backend_kwargs: Optional[dict] = None) -> "Array":
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.divide does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1 / arr2

    @classmethod
    def pow(cls, arr1: "Array", arr2: "Array", *, backend_kwargs: Optional[dict] = None) -> "Array":
        if backend_kwargs:
            raise TypeError(f"ArrayAPIBackend.pow does not accept keyword arguments: {sorted(backend_kwargs)}")
        return arr1**arr2

    @classmethod
    def take(
        cls,
        array: "Array",
        indices: "int | Array",
        dim: Optional[str | int] = None,
        *,
        backend_kwargs: Optional[dict] = None,
    ) -> "Array":
        import array_api_compat

        if not isinstance(dim, int):
            raise ValueError("Must provide `dim` as an integer")
        xp = array_api_compat.array_namespace(array)

        if hasattr(indices, "__iter__"):
            return xp.take(array, indices, axis=dim, **(backend_kwargs or {}))
        ret = xp.take(array, [indices], axis=dim, **(backend_kwargs or {}))
        return xp.squeeze(ret, axis=dim)
