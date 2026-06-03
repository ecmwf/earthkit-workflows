# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from typing import Any


def _common_keys(*dicts: dict[str, Any]) -> set[str]:
    """Return keys present in all provided dicts."""
    return set.intersection(*(set(d.keys()) for d in dicts))


def _delegate(name: str, *dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Delegate an operation to value-level backends for each common key.

    For each key present in all input dicts, calls ``backends.<name>``
    on the corresponding values, letting the dispatcher choose the
    correct backend for the value type.
    """
    # exceptional in-body import -- circular dependency with backends dispatcher
    from earthkit.workflows import backends

    keys = _common_keys(*dicts)
    return {k: getattr(backends, name)(*(d[k] for d in dicts), **kwargs) for k in keys}


class DictBackend:
    def mean(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _delegate("mean", *dicts, **kwargs)

    def std(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _delegate("std", *dicts, **kwargs)

    def max(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _delegate("max", *dicts, **kwargs)

    def min(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _delegate("min", *dicts, **kwargs)

    def sum(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _delegate("sum", *dicts, **kwargs)

    def prod(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _delegate("prod", *dicts, **kwargs)

    def var(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return _delegate("var", *dicts, **kwargs)

    def stack(*dicts: dict[str, Any], axis: int = 0, **kwargs: Any) -> dict[str, Any]:
        """Merge multiple dicts into one. Later dicts overwrite earlier keys."""
        result: dict[str, Any] = {}
        for d in dicts:
            result.update(d)
        return result

    def concat(*dicts: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Merge multiple dicts into one. Later dicts overwrite earlier keys."""
        result: dict[str, Any] = {}
        for d in dicts:
            result.update(d)
        return result

    def add(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return _delegate("add", a, b)

    def subtract(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return _delegate("subtract", a, b)

    def multiply(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return _delegate("multiply", a, b)

    def divide(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return _delegate("divide", a, b)

    def pow(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        return _delegate("pow", a, b)

    def take(array: dict[str, Any], indices: Any, *, dim: int, **kwargs: Any) -> dict[str, Any] | Any:
        if isinstance(indices, str):
            indices = [indices]
        if any(i not in array for i in indices):
            raise KeyError(f"One or more indices not found in array: {set(indices) - set(array.keys())}")
        if len(indices) == 1:
            return array[indices[0]]
        return {k: array[k] for k in indices}
