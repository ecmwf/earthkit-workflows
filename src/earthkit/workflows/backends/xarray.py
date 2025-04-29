# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import xarray as xr


class XArrayBackend:
    @staticmethod
    def multi_arg_function(
        name: str, *arrays: xr.DataArray | xr.Dataset, **method_kwargs
    ) -> xr.DataArray | xr.Dataset:
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
            arg = XArrayBackend.stack(*arrays, dim="**NEW**")
            method_kwargs["dim"] = "**NEW**"
        else:
            arg = arrays[0]

        return getattr(arg, name)(**method_kwargs)

    @staticmethod
    def two_arg_function(
        name: str,
        *arrays: xr.DataArray | xr.Dataset,
        keep_attrs: bool | str = False,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
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
        with xr.set_options(keep_attrs=keep_attrs):
            return getattr(np, name)(arrays[0], arrays[1], **method_kwargs)

    @staticmethod
    def mean(
        *arrays: xr.DataArray | xr.Dataset,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function("mean", *arrays, **method_kwargs)

    @staticmethod
    def std(
        *arrays: xr.DataArray | xr.Dataset,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function("std", *arrays, **method_kwargs)

    @staticmethod
    def min(
        *arrays: xr.DataArray | xr.Dataset,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function("min", *arrays, **method_kwargs)

    @staticmethod
    def max(
        *arrays: xr.DataArray | xr.Dataset,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function("max", *arrays, **method_kwargs)

    @staticmethod
    def sum(
        *arrays: xr.DataArray | xr.Dataset,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function("sum", *arrays, **method_kwargs)

    @staticmethod
    def prod(
        *arrays: xr.DataArray | xr.Dataset,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function("prod", *arrays, **method_kwargs)

    @staticmethod
    def var(
        *arrays: xr.DataArray | xr.Dataset,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function("var", *arrays, **method_kwargs)

    @staticmethod
    def concat(
        *arrays: xr.DataArray | xr.Dataset,
        dim: str,
        **method_kwargs: dict,
    ) -> xr.DataArray | xr.Dataset:
        if not np.any([dim in a.sizes for a in arrays]):
            raise ValueError(
                "Concat must be used on existing dimensions only. Try stack instead."
            )
        return xr.concat(arrays, dim=dim, **method_kwargs)  # type: ignore # xr/mypy dont coop

    @staticmethod
    def stack(
        *arrays: xr.DataArray | xr.Dataset,
        dim: str,
        axis: int = 0,
        **method_kwargs: dict,
    ) -> xr.DataArray | xr.Dataset:
        if np.any([dim in a.sizes for a in arrays]):
            raise ValueError(
                "Stack must be used on non-existing dimensions only. Try concat instead."
            )

        ret = xr.concat(arrays, dim=dim, **method_kwargs)  # type: ignore # xr/mypy dont coop
        dims = list(ret.sizes.keys())
        dim_index = dims.index(dim)
        if axis != dim_index:
            dims.pop(dim_index)
            ret = ret.transpose(*dims[:axis], dim, *dims[axis:])
        return ret

    @staticmethod
    def add(
        *arrays: xr.DataArray | xr.Dataset,
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "add", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    @staticmethod
    def subtract(
        *arrays: xr.DataArray | xr.Dataset,
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "subtract", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    @staticmethod
    def multiply(
        *arrays: xr.DataArray | xr.Dataset,
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "multiply", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    @staticmethod
    def pow(
        *arrays: xr.DataArray | xr.Dataset,
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "power", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    @staticmethod
    def divide(
        *arrays: xr.DataArray | xr.Dataset,
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "divide", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    @staticmethod
    def take(
        array,
        indices,
        *,
        dim: int | str,
        method: str = "isel",
        **method_kwargs,
    ):
        kwargs = {"drop": True}
        kwargs.update(method_kwargs)
        if isinstance(dim, int):
            dim = list(array.sizes.keys())[dim]

        return getattr(array, method)({dim: indices}, **kwargs)
