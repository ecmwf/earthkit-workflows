import functools
import xarray as xr
import numpy as np
from typing import Any

from .common import *


def concatenate(
    *arrays: list[xr.DataArray | xr.Dataset],
    **method_kwargs: dict,
) -> xr.DataArray | xr.Dataset:
    """
    Join along existing dimension in one of the inputs

    Parameters
    ----------
    arrays: list DataArrays or Datasets to apply function on
    method_kwargs: dict, kwargs for xarray.concat

    Return
    ------
    DataArray or Dataset
    """
    dim = method_kwargs.pop("dim")
    assert np.any([dim in a.dims for a in arrays])
    return xr.concat(arrays, dim=dim, **method_kwargs)


def stack(
    *arrays: list[xr.DataArray | xr.Dataset],
    axis: int | None = None,
    **method_kwargs: dict,
) -> xr.DataArray | xr.Dataset:
    """
    Join along new dimension in one of the inputs

    Parameters
    ----------
    arrays: list DataArrays or Datasets to apply function on
    axis: int | None, axis of new dimension if provided
    method_kwargs: dict, kwargs for xarray.concat

    Return
    ------
    DataArray or Dataset
    """
    dim = method_kwargs.pop("dim")
    assert not np.any([dim in a.dims for a in arrays])

    ret = xr.concat(arrays, dim=dim, **method_kwargs)
    if axis is not None and axis != 0:
        ret = ret.transpose(*ret.dims[1:axis], dim, *ret.dims[axis:])
    return ret


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
        arg = stack(*arrays, **stack_kwargs)
        method_kwargs["dim"] = stack_kwargs["dim"]
    else:
        arg = arrays[0]

    return method(arg, name, **method_kwargs)


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


mean = functools.partial(multi_arg_function, "mean")
std = functools.partial(multi_arg_function, "std")
maximum = functools.partial(multi_arg_function, "max")
minimum = functools.partial(multi_arg_function, "min")
sum = functools.partial(multi_arg_function, "sum")
product = functools.partial(multi_arg_function, "prod")
variance = functools.partial(multi_arg_function, "var")

subtract = functools.partial(two_arg_function, "subtract")
add = functools.partial(two_arg_function, "add")
multiply = functools.partial(two_arg_function, "multiply")
divide = functools.partial(two_arg_function, "divide")

take = np.take
