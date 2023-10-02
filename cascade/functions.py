import xarray as xr
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import functools

from meteokit import extreme as extreme
from earthkit.data import FieldList

from .grib import extreme_grib_headers, threshold_grib_headers, buffer_to_template


def _concatenate(*arrays) -> FieldList:
    # Combine earthkit data objects into one object
    return sum(arrays[1:], arrays[0])


def standardise_output(data):
    # Need to convert to numpy array as jax array is not yet supported
    # Also, nest the data to avoid problems with not finding geography attribute
    ret = np.asarray(data)
    if len(ret.shape) == 1:
        ret = np.asarray([ret])
    assert len(ret.shape) == 2
    return ret


def _multi_arg_function(func, *arrays):
    concat = _concatenate(*arrays).values
    assert len(concat) == len(arrays)
    res = getattr(jnp, func)(concat, axis=0)
    return FieldList.from_numpy(standardise_output(res), arrays[0][0].metadata())


def _norm(arr1, arr2):
    norm = jnp.linalg.norm(_concatenate(arr1, arr2).values, axis=0)
    return FieldList.from_numpy(standardise_output(norm), arr1.metadata())


def _two_arg_function(func, arr1, arr2, extract_keys: tuple = ()):
    arr2_meta = buffer_to_template(arr2.metadata()[0].get("buffer"))
    metadata = arr1.metadata()[0].override(
        {key: arr2_meta.get(key) for key in extract_keys}
    )
    res = getattr(jnp, func)(arr1.values, arr2.values)
    return FieldList.from_numpy(standardise_output(res), metadata)


_mean = functools.partial(_multi_arg_function, "mean")
_std = functools.partial(_multi_arg_function, "std")
_maximum = functools.partial(_multi_arg_function, "max")
_minimum = functools.partial(_multi_arg_function, "min")
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


def threshold(threshold_config: dict, arr: FieldList, edition: int = 1) -> FieldList:
    # Find all locations where np.nan appears as an ensemble value
    is_nan = jnp.isnan(arr.values)
    thesh = comp_str2func(threshold_config["comparison"])(
        arr.values, threshold_config["value"]
    )
    res = jnp.where(is_nan, jnp.nan, thesh)
    threshold_headers = threshold_grib_headers(edition, threshold_config)
    metadata = arr[0].metadata().override(threshold_headers)
    return FieldList.from_numpy(standardise_output(res), metadata)


def efi(
    clim: FieldList, ens: FieldList, eps: float, num_steps: int, control: bool = False
) -> FieldList:
    extreme_headers = extreme_grib_headers(clim, ens, num_steps)
    if control:
        extreme_headers.update({"marsType": "efic", "totalNumber": 1, "number": 0})
        metadata = ens[0].metadata().override(extreme_headers)
    else:
        extreme_headers.update({"marsType": "efi", "efiOrder": 0})
        metadata = ens[0].metadata().override(extreme_headers)
    res = extreme.efi(clim.values, ens.values, eps)
    return FieldList.from_numpy(standardise_output(res), metadata)


def sot(
    clim: FieldList, ens: FieldList, number: int, eps: float, num_steps: int
) -> FieldList:
    extreme_headers = extreme_grib_headers(clim, ens, num_steps)
    if number == 90:
        efi_order = 99
    elif number == 10:
        efi_order = 1
    else:
        raise Exception(
            "SOT value '{sot}' not supported in template! Only accepting 10 and 90"
        )
    metadata = (
        ens[0]
        .metadata()
        .override({**extreme_headers, "marsType": "sot", "efiOrder": efi_order})
    )
    res = extreme.sot(clim.values, ens.values, number, eps)
    return FieldList.from_numpy(standardise_output(res), metadata)


def wind_speed(arr: FieldList) -> FieldList:
    assert len(arr.values) == 2
    res = jnp.linalg.norm(arr.values, axis=0)
    return FieldList.from_numpy(standardise_output(res), arr[0].metadata())


def filter(
    comparison: str, threshold: float, arr1: FieldList, arr2: FieldList, replacement=0
) -> FieldList:
    condition = comp_str2func(comparison)(arr2.values, threshold)
    res = jnp.where(condition, replacement, arr1.values)
    return FieldList.from_numpy(standardise_output(res), arr1.metadata())
