# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import TypeAlias

import array_api_compat
from earthkit.data import FieldList
from earthkit.data.core.metadata import Metadata as ekdMetadata

from earthkit.workflows.backends import num_args


def standardise_output(data):
    # Also, nest the data to avoid problems with not finding geography attribute
    if len(data.shape) == 1:
        data = data.reshape((1, *data.shape))
    assert len(data.shape) == 2
    return data


def comp_str2func(array_module, comparison: str):
    if comparison == "<=":
        return array_module.less_equal
    if comparison == "<":
        return array_module.less
    if comparison == ">=":
        return array_module.greater_equal
    return array_module.greater


Metadata: TypeAlias = "dict | callable | None"


def resolve_metadata(metadata: Metadata, *args) -> dict:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    return metadata(*args)


def new_fieldlist(data, metadata: list[ekdMetadata], overrides: dict):
    if len(overrides) > 0:
        try:
            new_metadata = [
                metadata[x].override(overrides) for x in range(len(metadata))
            ]
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


class FieldListBackend:
    def _merge(*fieldlists: list[FieldList]):
        """Merge fieldlist elements into a single array. fieldlists with
        different number of fields must be concatenated, otherwise, the
        elements in each fieldlist are stacked along a new dimension
        """
        if len(fieldlists) == 1:
            return fieldlists[0].values

        values = [x.values for x in fieldlists]
        xp = array_api_compat.array_namespace(*values)
        return xp.asarray(values)

    def multi_arg_function(
        func: str, *arrays: list[FieldList], metadata: Metadata = None
    ) -> FieldList:
        merged_array = FieldListBackend._merge(*arrays)
        xp = array_api_compat.array_namespace(*merged_array)
        res = standardise_output(getattr(xp, func)(merged_array, axis=0))
        return new_fieldlist(
            res,
            [arrays[0][x].metadata() for x in range(len(res))],
            resolve_metadata(metadata, *arrays),
        )

    def two_arg_function(
        func: str, *arrays: FieldList, metadata: Metadata = None
    ) -> FieldList:
        # First argument must be FieldList
        assert isinstance(
            arrays[0], FieldList
        ), f"Expected FieldList type, got {type(arrays[0])}"
        val1 = arrays[0].values
        if isinstance(arrays[1], FieldList):
            val2 = arrays[1].values
            metadata = resolve_metadata(metadata, *arrays)
            xp = array_api_compat.array_namespace(val1, val2)
        else:
            val2 = arrays[1]
            metadata = resolve_metadata(metadata, arrays[0])
            xp = array_api_compat.array_namespace(val1)
        res = getattr(xp, func)(val1, val2)
        return new_fieldlist(
            res, [arrays[0][x].metadata() for x in range(len(res))], metadata
        )

    def mean(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multi_arg_function("mean", *arrays, metadata=metadata)

    def std(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multi_arg_function("std", *arrays, metadata=metadata)

    def min(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multi_arg_function("min", *arrays, metadata=metadata)

    def max(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multi_arg_function("max", *arrays, metadata=metadata)

    def sum(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multi_arg_function("sum", *arrays, metadata=metadata)

    def prod(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multi_arg_function("prod", *arrays, metadata=metadata)

    def var(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multi_arg_function("var", *arrays, metadata=metadata)

    def stack(*arrays: list[FieldList], axis: int = 0) -> FieldList:
        if axis != 0:
            raise ValueError("Can not stack FieldList along axis != 0")
        assert all(
            [len(x) == 1 for x in arrays]
        ), "Can not stack FieldLists with more than one element, use concat"
        return FieldListBackend.concat(*arrays)

    def add(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.two_arg_function("add", *arrays, metadata=metadata)

    def subtract(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.two_arg_function("subtract", *arrays, metadata=metadata)

    @num_args(2)
    def diff(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.multiply(
            FieldListBackend.subtract(*arrays, metadata=metadata), -1
        )

    def multiply(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.two_arg_function("multiply", *arrays, metadata=metadata)

    def divide(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.two_arg_function("divide", *arrays, metadata=metadata)

    def pow(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        return FieldListBackend.two_arg_function("pow", *arrays, metadata=metadata)

    def concat(*arrays: list[FieldList]) -> FieldList:
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

    def take(
        array: FieldList,
        indices: int | tuple,
        *,
        dim: int | str,
        method: str = "slice",
        **kwargs,
    ) -> FieldList:
        if method == "slice":
            if dim != 0:
                raise ValueError("Can not slice from FieldList along dim != 0")
            if isinstance(indices, int):
                indices = [indices]
            ret = array[indices]
        else:
            if not isinstance(dim, str):
                raise ValueError(
                    "To perform isel/sel on FieldList, dim must be a string"
                )
            if method == "isel":
                ret = array.isel(**{dim: indices}, **kwargs)
            elif method == "sel":
                ret = array.sel(**{dim: indices}, **kwargs)
            else:
                raise ValueError(f"Invalid method {method}")

        return FieldList.from_array(ret.values, ret.metadata())

    def norm(*arrays: list[FieldList], metadata: Metadata = None) -> FieldList:
        merged_array = FieldListBackend._merge(*arrays)
        xp = array_api_compat.array_namespace(merged_array)
        norm = standardise_output(xp.sqrt(xp.sum(xp.pow(merged_array, 2), axis=0)))
        return new_fieldlist(
            norm,
            [arrays[0][x].metadata() for x in range(len(norm))],
            resolve_metadata(metadata, *arrays),
        )

    def filter(
        arr1: FieldList,
        arr2: FieldList,
        comparison: str,
        threshold: float,
        *,
        replacement: float = 0,
        metadata: Metadata = None,
    ) -> FieldList:
        xp = array_api_compat.array_namespace(arr1.values, arr2.values)
        condition = comp_str2func(xp, comparison)(arr2.values, threshold)
        res = xp.where(condition, replacement, arr1.values)
        return new_fieldlist(
            res, arr1.metadata(), resolve_metadata(metadata, arr1, arr2)
        )

    def set_metadata(data: FieldList, metadata: dict) -> FieldList:
        return new_fieldlist(data.values, data.metadata(), metadata)
