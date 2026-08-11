# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Optional

from .base import Backend


class XArrayBackend(Backend):
    @staticmethod
    def multi_arg_function(name: str, *arrays, **method_kwargs):
        """Apply named function on DataArrays or Datasets. If only a single
        DataArrays or Datasetst then function is applied
        along an dimension specified in method_kwargs. If multiple  DataArrays
        or Datasets then these are first stacked before function is applied on the
        stack

        Parameters
        ----------
        name: str, name of function to apply
        arrays: list DataArrays or Datasets to apply function on
        method_kwargs: dict, kwargs for named function

        Return
        ------
        DataArray or Dataset
        """
        if len(arrays) > 1:
            arg = XArrayBackend.stack(*arrays, backend_kwargs={"dim": "**NEW**"})
            method_kwargs["dim"] = "**NEW**"
        else:
            arg = arrays[0]

        return getattr(arg, name)(**method_kwargs)

    @staticmethod
    def two_arg_function(
        name: str,
        arr1,
        arr2,
        *,
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        """Apply named function in numpy on list of DataArrays or Datasets.

        Parameters
        ----------
        name: str, name of function to apply
        arrays: list DataArrays or Datasets to apply function on
        keep_attrs: bool or str, sets xarray options regarding keeping attributes in the
        computation. If "default", then attributes are only kept in unambiguous cases.

        Return
        ------
        DataArray or Dataset

        Raises
        ------
        AssertionError if more than two DataArrays or Datasets are passed as inputs
        """
        import numpy as np
        import xarray as xr

        with xr.set_options(keep_attrs=keep_attrs):
            return getattr(np, name)(arr1, arr2, **method_kwargs)

    @staticmethod
    def mean(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.multi_arg_function("mean", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def std(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.multi_arg_function("std", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def min(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.multi_arg_function("min", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def max(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.multi_arg_function("max", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def sum(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.multi_arg_function("sum", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def prod(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.multi_arg_function("prod", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def var(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.multi_arg_function("var", *arrays, **(backend_kwargs or {}))

    @staticmethod
    def concat(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        import numpy as np
        import xarray as xr

        backend_kwargs = (backend_kwargs or {}).copy()
        if "dim" not in backend_kwargs:
            raise TypeError("XArrayBackend.concat requires a 'dim' to be specified in backend_kwargs")
        dim = backend_kwargs.pop("dim")
        if not np.any([dim in a.sizes for a in arrays]):
            raise ValueError("Concat must be used on existing dimensions only. Try stack instead.")
        return xr.concat(arrays, dim=dim, **(backend_kwargs or {}))  # type: ignore # xr/mypy dont coop

    @staticmethod
    def stack(
        *arrays,
        backend_kwargs: Optional[dict] = None,
    ):
        import numpy as np
        import xarray as xr

        backend_kwargs = (backend_kwargs or {}).copy()
        if "dim" not in backend_kwargs:
            raise TypeError("XArrayBackend.stack requires a 'dim' to be specified in backend_kwargs")
        dim = backend_kwargs.pop("dim")
        axis = backend_kwargs.pop("axis", 0)

        if np.any([dim in a.sizes for a in arrays]):
            raise ValueError("Stack must be used on non-existing dimensions only. Try concat instead.")

        ret = xr.concat(arrays, dim=dim, **backend_kwargs)  # type: ignore # xr/mypy dont coop
        dims = list(ret.sizes.keys())
        dim_index = dims.index(dim)
        if axis != dim_index:
            dims.pop(dim_index)
            ret = ret.transpose(*dims[:axis], dim, *dims[axis:])
        return ret

    @staticmethod
    def add(
        arr1,
        arr2,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.two_arg_function("add", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def subtract(
        arr1,
        arr2,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.two_arg_function("subtract", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def multiply(
        arr1,
        arr2,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.two_arg_function("multiply", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def pow(
        arr1,
        arr2,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.two_arg_function("power", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def divide(
        arr1,
        arr2,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
        return XArrayBackend.two_arg_function("divide", arr1, arr2, **(backend_kwargs or {}))

    @staticmethod
    def take(
        array,
        indices,
        dim: Optional[int | str] = None,
        *,
        backend_kwargs: Optional[dict] = None,
    ):
        if dim is None:
            raise TypeError("XArrayBackend.take requires a 'dim' to be specified")
        kwargs: dict[str, Any] = {"drop": True}
        kwargs.update(backend_kwargs or {})
        method: str = kwargs.pop("method", "isel")
        if isinstance(dim, int):
            dim = list(array.sizes.keys())[dim]

        return getattr(array, method)({dim: indices}, **kwargs)
