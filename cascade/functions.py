import functools
import array_api_compat

from meteokit import extreme
from meteokit.stats import iter_quantiles
from earthkit.data import FieldList
from earthkit.data.sources.numpy_list import NumpyFieldList

from .grib import extreme_grib_headers, threshold_grib_headers
from .patch import PatchModule


def concatenate(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
    # Combine earthkit data objects into one object
    return sum(arrays[1:], arrays[0])


def standardise_output(data):
    # Also, nest the data to avoid problems with not finding geography attribute
    ret = data
    if len(ret.shape) == 1:
        ret = ret.reshape((1, *data.shape))
    assert len(ret.shape) == 2
    return ret


def multi_arg_function(func: str, *arrays: list[NumpyFieldList]) -> NumpyFieldList:
    if len(arrays) == 1:
        concat = arrays[0].values
    else:
        concat = concatenate(*arrays).values
        assert len(concat) == len(arrays)

    xp = array_api_compat.array_namespace(concat)
    res = getattr(xp, func)(concat, axis=0)
    return FieldList.from_numpy(standardise_output(res), arrays[0][0].metadata())


def norm(*arrays: list[NumpyFieldList]) -> NumpyFieldList:
    vals = [x.values for x in arrays]
    xp = array_api_compat.array_namespace(*vals)
    if len(vals) == 1:
        # Assume fields to compute norm of are nested in single field list
        vals = vals[0]
    assert len(vals) == 2, f"Expected 2 fields for norm, received {len(vals)}"
    norm = xp.sqrt(vals[0] ** 2 + vals[1] ** 2)
    return FieldList.from_numpy(standardise_output(norm), arrays[0][0].metadata())


def two_arg_function(
    func: str, *arrays: NumpyFieldList, extract_keys: tuple = ()
) -> NumpyFieldList:
    vals = [x.values for x in arrays]
    xp = array_api_compat.array_namespace(*vals)
    if len(vals) == 1:
        # Assume fields to compute norm of are nested in single field list
        vals = vals[0]
        arr2_meta = arrays[0][1].metadata().buffer_to_metadata()
    else:
        arr2_meta = arrays[1][0].metadata().buffer_to_metadata()
    assert (
        len(vals) == 2
    ), f"Expected 2 fields for two_arg_functions@{func}, received {len(vals)}"
    metadata = (
        arrays[0][0]
        .metadata()
        .override({key: arr2_meta.get(key) for key in extract_keys})
    )
    res = getattr(xp, func)(vals[0], vals[1])
    return FieldList.from_numpy(standardise_output(res), metadata)


mean = functools.partial(multi_arg_function, "mean")
std = functools.partial(multi_arg_function, "std")
maximum = functools.partial(multi_arg_function, "max")
minimum = functools.partial(multi_arg_function, "min")
subtract = functools.partial(two_arg_function, "subtract")
add = functools.partial(two_arg_function, "add")
multiply = functools.partial(two_arg_function, "multiply")
divide = functools.partial(two_arg_function, "divide")


def comp_str2func(array_module, comparison: str):
    if comparison == "<=":
        return array_module.less_equal
    if comparison == "<":
        return array_module.less
    if comparison == ">=":
        return array_module.greater_equal
    return array_module.greater


def threshold(
    threshold_config: dict, arr: NumpyFieldList, edition: int = 1
) -> NumpyFieldList:
    xp = array_api_compat.array_namespace(arr.values)
    # Find all locations where np.nan appears as an ensemble value
    is_nan = xp.isnan(arr.values)
    thesh = comp_str2func(xp, threshold_config["comparison"])(
        arr.values, threshold_config["value"]
    )
    res = xp.where(is_nan, xp.nan, thesh)
    threshold_headers = threshold_grib_headers(edition, threshold_config)
    metadata = arr[0].metadata().override(threshold_headers)
    return FieldList.from_numpy(standardise_output(res), metadata)


def efi(
    clim: NumpyFieldList,
    ens: NumpyFieldList,
    eps: float,
    num_steps: int,
    control: bool = False,
) -> NumpyFieldList:
    extreme_headers = extreme_grib_headers(clim, ens, num_steps)
    if control:
        extreme_headers.update({"marsType": "efic", "totalNumber": 1, "number": 0})
        metadata = ens[0].metadata().override(extreme_headers)
    else:
        extreme_headers.update({"marsType": "efi", "efiOrder": 0})
        metadata = ens[0].metadata().override(extreme_headers)

    xp = array_api_compat.array_namespace(ens.values, clim.values)
    with PatchModule(extreme, "numpy", xp):
        res = extreme.efi(clim.values, ens.values, eps)
    return FieldList.from_numpy(standardise_output(res), metadata)


def sot(
    clim: NumpyFieldList, ens: NumpyFieldList, number: int, eps: float, num_steps: int
) -> NumpyFieldList:
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

    xp = array_api_compat.array_namespace(ens.values, clim.values)
    with PatchModule(extreme, "numpy", xp):
        res = extreme.sot(clim.values, ens.values, number, eps)
    return FieldList.from_numpy(standardise_output(res), metadata)


def quantiles(ens: NumpyFieldList, quantile: float) -> NumpyFieldList:
    xp = array_api_compat.array_namespace(ens.values)
    with PatchModule(extreme, "numpy", xp):
        res = list(iter_quantiles(ens.values, [quantile], method="numpy"))[0]
    return FieldList.from_numpy(standardise_output(res), ens[0].metadata())


def filter(
    comparison: str,
    threshold: float,
    arr1: NumpyFieldList,
    arr2: NumpyFieldList,
    replacement=0,
) -> NumpyFieldList:
    xp = array_api_compat.array_namespace(arr1.values, arr2.values)
    condition = comp_str2func(xp, comparison)(arr2.values, threshold)
    res = xp.where(condition, replacement, arr1.values)
    return FieldList.from_numpy(standardise_output(res), arr1.metadata())
