# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import TYPE_CHECKING, Callable, Optional, Sequence, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from earthkit.data import FieldList

from .base import Backend

Indexer: TypeAlias = int | str | Sequence[int] | Sequence[str]
Metadata: TypeAlias = dict | Callable | None


def standardise_output(data: "np.ndarray") -> "np.ndarray":
    # Also, nest the data to avoid problems with not finding geography attribute
    if len(data.shape) == 1:
        data = data.reshape((1, *data.shape))
    assert len(data.shape) == 2
    return data


def resolve_metadata(metadata: Metadata, *args) -> dict:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    return metadata(*args)


def new_fieldlist(data: "np.ndarray", metadata: list, overrides: dict) -> "FieldList":
    from earthkit.data import FieldList

    if len(overrides) > 0:
        try:
            new_metadata = [metadata[x].override(overrides) for x in range(len(metadata))]
            return FieldList.from_array(
                standardise_output(data),
                new_metadata,
            )
        except Exception as e:
            print(
                "Error setting metadata",
                overrides,
                "On data with:",
                list(map(lambda x: x.dump(), metadata)),
            )
            print(e)
    return FieldList.from_array(standardise_output(data), metadata)


class FieldListBackend(Backend):
    def _merge(*fieldlists: "FieldList") -> "FieldList":
        """Merge fieldlist elements into a single array. fieldlists with
        different number of fields must be concatenated, otherwise, the
        elements in each fieldlist are stacked along a new dimension
        """
        import array_api_compat

        if len(fieldlists) == 1:
            return fieldlists[0].values

        values = [x.values for x in fieldlists]
        xp = array_api_compat.array_namespace(*values)
        return xp.asarray(values)

    @classmethod
    def multi_arg_function(cls, func: str, *arrays: "FieldList", metadata: Metadata = None) -> "FieldList":
        import array_api_compat

        merged_array = FieldListBackend._merge(*arrays)
        xp = array_api_compat.array_namespace(*merged_array)
        is_nan = xp.isnan(merged_array).any(axis=0)
        res = xp.where(is_nan, xp.nan, standardise_output(getattr(xp, func)(merged_array, axis=0)))
        return new_fieldlist(
            res,
            [arrays[0][x].metadata() for x in range(len(res))],
            resolve_metadata(metadata, *arrays),
        )

    @classmethod
    def two_arg_function(cls, func: str, arr1: "FieldList", arr2: "FieldList | np.ndarray", metadata: Metadata = None) -> "FieldList":
        import array_api_compat
        from earthkit.data import FieldList

        assert isinstance(arr1, FieldList), f"Expected FieldList type, got {type(arr1)}"
        val1 = arr1.values
        if isinstance(arr2, FieldList):
            val2 = arr2.values
            metadata = resolve_metadata(metadata, arr1, arr2)
            xp = array_api_compat.array_namespace(val1, val2)
        else:
            val2 = arr2
            metadata = resolve_metadata(metadata, arr1)
            xp = array_api_compat.array_namespace(val1)
        res = getattr(xp, func)(val1, val2)
        return new_fieldlist(res, [arr1[x].metadata() for x in range(len(res))], metadata)

    @classmethod
    def mean(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multi_arg_function("mean", *arrays, **(backend_kwargs or {}))

    @classmethod
    def std(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multi_arg_function("std", *arrays, **(backend_kwargs or {}))

    @classmethod
    def min(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multi_arg_function("min", *arrays, **(backend_kwargs or {}))

    @classmethod
    def max(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multi_arg_function("max", *arrays, **(backend_kwargs or {}))

    @classmethod
    def sum(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multi_arg_function("sum", *arrays, **(backend_kwargs or {}))

    @classmethod
    def prod(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multi_arg_function("prod", *arrays, **(backend_kwargs or {}))

    @classmethod
    def var(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multi_arg_function("var", *arrays, **(backend_kwargs or {}))

    @classmethod
    def stack(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        if backend_kwargs:
            raise TypeError(f"FieldListBackend.stack does not accept keyword arguments: {sorted(backend_kwargs)}")
        assert all([len(x) == 1 for x in arrays]), "Can not stack FieldLists with more than one element, use concat"
        return FieldListBackend.concat(*arrays)

    @classmethod
    def add(cls, arr1: "FieldList", arr2: "FieldList", *, backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.two_arg_function("add", arr1, arr2, **(backend_kwargs or {}))

    @classmethod
    def subtract(cls, arr1: "FieldList", arr2: "FieldList", *, backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.two_arg_function("subtract", arr1, arr2, **(backend_kwargs or {}))

    @classmethod
    def diff(cls, arr1: "FieldList", arr2: "FieldList", *, backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.multiply(
            FieldListBackend.subtract(arr1, arr2, backend_kwargs=backend_kwargs),
            -1,  # type: ignore[arg-type]
        )

    @classmethod
    def multiply(cls, arr1: "FieldList", arr2: "FieldList", *, backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.two_arg_function("multiply", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def divide(arr1: "FieldList", arr2: "FieldList", *, backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.two_arg_function("divide", arr1, arr2, **(backend_kwargs or {}))

    @classmethod
    def divide(cls, arr1: "FieldList", arr2: "FieldList", *, backend_kwargs: Optional[dict] = None) -> "FieldList":
        return FieldListBackend.two_arg_function("pow", arr1, arr2, **(backend_kwargs or {}))

    @classmethod
    def concat(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        """Concatenates the list of fields inside each FieldList into a single
        FieldList object

        Parameters
        ----------
        arrays: list[FieldList]
            FieldList instances to whose fields are to be concatenated

        Return
        ------
        FieldList
            Contains all fields inside the input field lists
        """
        ret = sum(arrays[1:], arrays[0])
        return ret

    @classmethod
    def take(
        cls,
        array: "FieldList",
        indices: Indexer,
        dim: Optional[int | str] = None,
        *,
        backend_kwargs: Optional[dict] = None,
    ) -> "FieldList":
        from earthkit.data import FieldList

        backend_kwargs = (backend_kwargs or {}).copy()
        method = backend_kwargs.pop("method", "slice")
        if method == "slice":
            if dim is not None and dim != 0:
                raise ValueError("Can not slice from FieldList along dim != 0")
            if isinstance(indices, int):
                indices = [indices]  # type: ignore[assignment]
            ret = array[indices]
        else:
            if not isinstance(dim, str):
                raise ValueError("To perform isel/sel on FieldList, dim must be a string")
            if method == "isel":
                ret = array.isel(**{dim: indices}, **backend_kwargs)
            elif method == "sel":
                ret = array.sel(**{dim: indices}, **backend_kwargs)
            else:
                raise ValueError(f"Invalid method {method}")

        if len(ret) == 0:
            raise ValueError(f"Take along dim {dim} resulted in empty fieldlist: indices {indices}, method {method}")
        return FieldList.from_array(ret.values, ret.metadata())

    @classmethod
    def norm(cls, *arrays: "FieldList", backend_kwargs: Optional[dict] = None) -> "FieldList":
        backend_kwargs = (backend_kwargs or {}).copy()
        metadata = backend_kwargs.pop("metadata", None)
        import array_api_compat

        merged_array = FieldListBackend._merge(*arrays)
        xp = array_api_compat.array_namespace(merged_array)
        norm = standardise_output(xp.sqrt(xp.sum(xp.pow(merged_array, 2), axis=0)))
        return new_fieldlist(
            norm,
            [arrays[0][x].metadata() for x in range(len(norm))],
            resolve_metadata(metadata, *arrays),
        )

    @classmethod
    def filter(
        cls,
        arr1: "FieldList",
        mask: "FieldList",
        replacement: float = 0,
        *,
        backend_kwargs: Optional[dict] = None,
    ) -> "FieldList":
        import array_api_compat

        backend_kwargs = (backend_kwargs or {}).copy()
        metadata = backend_kwargs.pop("metadata", None)
        xp = array_api_compat.array_namespace(arr1.values, mask.values)
        res = xp.where(mask.values, replacement, arr1.values)
        return new_fieldlist(res, arr1.metadata(), resolve_metadata(metadata, arr1, mask))

    @classmethod
    def set_metadata(cls, data: "FieldList", metadata: dict) -> "FieldList":
        return new_fieldlist(data.values, data.metadata(), metadata)
