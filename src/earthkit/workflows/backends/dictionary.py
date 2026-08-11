# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from typing import Any, Optional

from .base import Backend


def _common_keys(*dicts: dict[str, Any]) -> list[str]:
    """Return keys present in all provided dicts, preserving the first dict's order."""
    if not dicts:
        return []

    common = set(dicts[0].keys())
    for d in dicts[1:]:
        common.intersection_update(d.keys())

    return [k for k in dicts[0].keys() if k in common]


def _delegate(name: str, *dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Delegate an operation to value-level backends for each common key.

    For each key present in all input dicts, calls ``backends.<name>``
    on the corresponding values, letting the dispatcher choose the
    correct backend for the value type.
    """
    # exceptional in-body import -- circular dependency with backends dispatcher
    from earthkit.workflows.backends import method

    keys = _common_keys(*dicts)
    return {k: method(name, *(d[k] for d in dicts), **kwargs) for k in keys}


class DictBackend(Backend):
    @staticmethod
    def mean(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("mean", *dicts, backend_kwargs=backend_kwargs)

    @staticmethod
    def std(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("std", *dicts, backend_kwargs=backend_kwargs)

    @staticmethod
    def max(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("max", *dicts, backend_kwargs=backend_kwargs)

    @staticmethod
    def min(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("min", *dicts, backend_kwargs=backend_kwargs)

    @staticmethod
    def sum(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("sum", *dicts, backend_kwargs=backend_kwargs)

    @staticmethod
    def prod(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("prod", *dicts, backend_kwargs=backend_kwargs)

    @staticmethod
    def var(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("var", *dicts, backend_kwargs=backend_kwargs)

    @staticmethod
    def stack(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        """Merge multiple dicts into one. Later dicts overwrite earlier keys."""
        if backend_kwargs:
            raise TypeError(f"DictBackend.stack does not accept keyword arguments: {sorted(backend_kwargs)}")
        result: dict[str, Any] = {}
        for d in dicts:
            result.update(d)
        return result

    @staticmethod
    def concat(*dicts: dict[str, Any], backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        """Merge multiple dicts into one. Later dicts overwrite earlier keys."""
        if backend_kwargs:
            raise TypeError(f"DictBackend.concat does not accept keyword arguments: {sorted(backend_kwargs)}")
        result: dict[str, Any] = {}
        for d in dicts:
            result.update(d)
        return result

    @staticmethod
    def add(arr1: dict[str, Any], arr2: dict[str, Any], *, backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("add", arr1, arr2, backend_kwargs=backend_kwargs)

    @staticmethod
    def subtract(arr1: dict[str, Any], arr2: dict[str, Any], *, backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("subtract", arr1, arr2, backend_kwargs=backend_kwargs)

    @staticmethod
    def multiply(arr1: dict[str, Any], arr2: dict[str, Any], *, backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("multiply", arr1, arr2, backend_kwargs=backend_kwargs)

    @staticmethod
    def divide(arr1: dict[str, Any], arr2: dict[str, Any], *, backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("divide", arr1, arr2, backend_kwargs=backend_kwargs)

    @staticmethod
    def pow(arr1: dict[str, Any], arr2: dict[str, Any], *, backend_kwargs: Optional[dict] = None) -> dict[str, Any]:
        return _delegate("pow", arr1, arr2, backend_kwargs=backend_kwargs)

    @staticmethod
    def take(
        array: dict[str, Any], indices: Any, dim: Optional[str | int] = None, *, backend_kwargs: Optional[dict] = None
    ) -> dict[str, Any] | Any:
        if dim is not None:
            raise TypeError("DictBackend.take does not support the 'dim' argument")
        if backend_kwargs:
            raise TypeError(f"DictBackend.take does not accept keyword arguments: {sorted(backend_kwargs)}")
        if isinstance(indices, str):
            indices = [indices]
        if any(i not in array for i in indices):
            raise KeyError(f"One or more indices not found in array: {set(indices) - set(array.keys())}")
        if len(indices) == 1:
            return array[indices[0]]
        return {k: array[k] for k in indices}
