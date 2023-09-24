import xarray as xr
import numpy as np
import jax.numpy as jnp
import functools

from meteokit import extreme as extreme
from earthkit.data import FieldList


def _concatenate(*arrays):
    # Combine earthkit data objects into one object
    return sum(arrays[1:], arrays[0])


def _multi_arg_function(func, *arrays):
    concat = _concatenate(*arrays).values
    assert len(concat) == len(arrays)
    res = getattr(jnp, func)(concat, axis=0)
    return FieldList.from_numpy(np.asarray(res), arrays[0][0].metadata())


def _norm(arr1, arr2):
    norm = jnp.linalg.norm(_concatenate([arr1, arr2]).values, axis=0)
    return FieldList.from_numpy(np.asarray(norm), arr1.metadata())


def _two_arg_function(func, arr1, arr2):
    res = getattr(jnp, func)(arr1.values, arr2.values)
    return FieldList.from_numpy(np.asarray(res), arr1.metadata())


_mean = functools.partial(_multi_arg_function, "mean")
_std = functools.partial(_multi_arg_function, "std")
_maximum = functools.partial(_multi_arg_function, "maximum")
_minimum = functools.partial(_multi_arg_function, "minimum")
_subtract = functools.partial(_two_arg_function, "subtract")
_add = functools.partial(_two_arg_function, "add")
_multiply = functools.partial(_two_arg_function, "multiply")
_divide = functools.partial(_two_arg_function, "divide")


def comp_str2func(comparison: str):
    if comparison == "<=":
        return jnp.less_equal
    if comparison == "<":
        return jnp.less
    if comparison == ">=":
        return jnp.greater_equal
    return jnp.greater


def threshold(comparison: str, threshold: float, arr):
    res = comp_str2func(comparison)(arr.values, threshold)
    return FieldList.from_numpy(np.asarray(res), arr.metadata())


def efi(clim, ens, eps: float):
    res = extreme.efi(clim.values, ens.values, eps)
    return FieldList.from_numpy(res, ens.metadata())


def sot(clim, ens, number: int, eps: float):
    res = extreme.sot(clim.values, ens.values, number, eps)
    return FieldList.from_numpy(res, ens.metadata())


def wind_speed(arr):
    assert len(arr.values) == 2
    res = jnp.linalg.norm(arr.values, axis=0)
    return FieldList.from_numpy(np.asarray(res), arr[0].metadata())


def filter(comparison, threshold, arr1, arr2, replacement=0):
    condition = comp_str2func(comparison)(arr2.values, threshold)
    res = jnp.where(condition, replacement, arr1.values)
    return FieldList.from_numpy(np.asarray(res), arr1.metadata())
