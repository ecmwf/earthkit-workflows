import xarray as xr
import numpy as np
from typing import Any

from .base import BaseBackend


class XArrayBackend(BaseBackend):
    def multi_arg_function(
        name: str,
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply named function on DataArrays or Datasets. If only a single
        DataArrays or Datasetst hen function is applied
        along an dimension specified in method_kwargs. If multiple  DataArrays
        or Datasets then these are first stacked before function is applied on the
        stack

        Parameters
        ----------
        name: str, name of function to apply
        arrays: list DataArrays or Datasets to apply function on
        stack_kwargs: dict, kwargs for stack
        method_kwargs: dict, kwargs for named function

        Return
        ------
        DataArray or Dataset
        """
        if method_kwargs is None:
            method_kwargs = {}

        if len(arrays) > 1:
            if stack_kwargs is None:
                stack_kwargs = {}
            stack_kwargs.setdefault("dim", "**NEW**")
            arg = XArrayBackend.stack(*arrays, **stack_kwargs)
            method_kwargs["dim"] = stack_kwargs["dim"]
        else:
            arg = arrays[0]

        return getattr(arg, name)(**method_kwargs)

    def two_arg_function(
        name: str,
        *arrays: list[xr.DataArray | xr.Dataset],
        keep_attrs: bool | str = False,
        **method_kwargs,
    ) -> xr.DataArray | xr.Dataset:
        """
        Apply named function in numpy on list of DataArrays or Datasets.

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
        assert (
            len(arrays) == 2
        ), f"two_arg_function {name} expects two input arguments, got {len(arrays)}"

        with xr.set_options(keep_attrs=keep_attrs):
            return getattr(np, name)(arrays[0], arrays[1], **method_kwargs)

    def mean(
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function(
            "mean", *arrays, stack_kwargs=stack_kwargs, method_kwargs=method_kwargs
        )

    def std(
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function(
            "std", *arrays, stack_kwargs=stack_kwargs, method_kwargs=method_kwargs
        )

    def min(
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function(
            "min", *arrays, stack_kwargs=stack_kwargs, method_kwargs=method_kwargs
        )

    def max(
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function(
            "max", *arrays, stack_kwargs=stack_kwargs, method_kwargs=method_kwargs
        )

    def sum(
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function(
            "sum", *arrays, stack_kwargs=stack_kwargs, method_kwargs=method_kwargs
        )

    def prod(
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function(
            "prod", *arrays, stack_kwargs=stack_kwargs, method_kwargs=method_kwargs
        )

    def var(
        *arrays: list[xr.DataArray | xr.Dataset],
        stack_kwargs: dict[str, Any] | None = None,
        method_kwargs: dict[str, Any] | None = None,
    ) -> xr.DataArray | xr.Dataset:
        return XArrayBackend.multi_arg_function(
            "var", *arrays, stack_kwargs=stack_kwargs, method_kwargs=method_kwargs
        )

    def concat(
        *arrays: list[xr.DataArray | xr.Dataset],
        **method_kwargs: dict,
    ) -> xr.DataArray | xr.Dataset:
        dim = method_kwargs.pop("dim")
        assert np.any([dim in a.dims for a in arrays])
        return xr.concat(arrays, dim=dim, **method_kwargs)

    def stack(
        *arrays: list[xr.DataArray | xr.Dataset],
        axis: int | None = None,
        **method_kwargs: dict,
    ) -> xr.DataArray | xr.Dataset:
        dim = method_kwargs.pop("dim")
        assert not np.any([dim in a.dims for a in arrays])

        ret = xr.concat(arrays, dim=dim, **method_kwargs)
        if axis is not None and axis != 0:
            ret = ret.transpose(*ret.dims[1:axis], dim, *ret.dims[axis:])
        return ret

    def add(
        *arrays: list[xr.DataArray | xr.Dataset],
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "add", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    def subtract(
        *arrays: list[xr.DataArray | xr.Dataset],
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "subtract", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    def multiply(
        *arrays: list[xr.DataArray | xr.Dataset],
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "multiply", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    def divide(
        *arrays: list[xr.DataArray | xr.Dataset],
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "divide", *arrays, keep_attrs=keep_attrs, **method_kwargs
        )

    def diff(
        *arrays: list[xr.DataArray | xr.Dataset],
        keep_attrs: bool | str = False,
        **method_kwargs,
    ):
        return XArrayBackend.two_arg_function(
            "subtract", arrays[1], arrays[0], keep_attrs=keep_attrs, **method_kwargs
        )

    def take(array, indices, *, axis: int, **kwargs):
        if hasattr(indices, "__iter__"):
            return np.take(array, indices, axis=axis, **kwargs)
        ret = np.take(array, [indices], axis=axis, **kwargs)
        return np.squeeze(ret, axis=axis)
