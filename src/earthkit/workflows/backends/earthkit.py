# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Callable, Optional, TypeAlias

from .base import Backend


def standardise_output(data):
    # Also, nest the data to avoid problems with not finding geography attribute
    if len(data.shape) == 1:
        data = data.reshape((1, *data.shape))
    assert len(data.shape) == 2
    return data


Metadata: TypeAlias = dict | Callable | None


def resolve_metadata(metadata: Metadata, *args) -> dict:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    return metadata(*args)


def new_fieldlist(data, metadata: list, overrides: dict):
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
    def _merge(*fieldlists):
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

    def multi_arg_function(func: str, *arrays, metadata: Metadata = None):
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

    def two_arg_function(func: str, arr1, arr2, metadata: Metadata = None):
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

    @staticmethod
    def mean(*arrays, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multi_arg_function("mean", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def std(*arrays, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multi_arg_function("std", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def min(*arrays, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multi_arg_function("min", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def max(*arrays, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multi_arg_function("max", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def sum(*arrays, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multi_arg_function("sum", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def prod(*arrays, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multi_arg_function("prod", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def var(*arrays, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multi_arg_function("var", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def stack(*arrays, backend_kwargs: Optional[dict] = None):
        if backend_kwargs:
            raise TypeError(f"FieldListBackend.stack does not accept keyword arguments: {sorted(backend_kwargs)}")
        assert all([len(x) == 1 for x in arrays]), "Can not stack FieldLists with more than one element, use concat"
        return FieldListBackend.concat(*arrays)

    @staticmethod
    def add(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.two_arg_function("add", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def subtract(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.two_arg_function("subtract", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def diff(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.multiply(
            FieldListBackend.subtract(arr1, arr2, backend_kwargs=backend_kwargs),
            -1,  # type: ignore[arg-type]
        )

    @staticmethod
    def multiply(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.two_arg_function("multiply", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def divide(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.two_arg_function("divide", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def pow(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        return FieldListBackend.two_arg_function("pow", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def concat(*arrays, backend_kwargs: Optional[dict] = None):
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

    @staticmethod
    def take(
        array,
        indices: int | tuple,
        dim: Optional[int | str] = None,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
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

    def norm(*arrays, backend_kwargs: Optional[dict] = None):
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

    @staticmethod
    def filter(
        arr1,
        mask,
        replacement: float = 0,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
        import array_api_compat

        backend_kwargs = (backend_kwargs or {}).copy()
        metadata = backend_kwargs.pop("metadata", None)
        xp = array_api_compat.array_namespace(arr1.values, mask.values)
        res = xp.where(mask.values, replacement, arr1.values)
        return new_fieldlist(res, arr1.metadata(), resolve_metadata(metadata, arr1, mask))

    @staticmethod
    def set_metadata(data, metadata: dict):
        return new_fieldlist(data.values, data.metadata(), metadata)
