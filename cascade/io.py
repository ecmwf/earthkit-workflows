import xarray as xr

from pproc.common.io import (
    target_from_location,
    write_grib,
    fdb_read,
    fdb,
    FileTarget,
    FileSetTarget,
)


def retrieve(request: dict):
    interpolation_keys = request.pop("interpolate", None)
    return fdb_read(fdb(), request, interpolation_keys)


def write(loc: str, data: xr.DataArray, grib_sets: dict):
    target = target_from_location(loc)
    if isinstance(target, (FileTarget, FileSetTarget)):
        # Allows file to be appended on each write call
        target.enable_recovery()
    template = data.attrs["grib_template"].copy()
    template.set(grib_sets)
    write_grib(target, template, data.data)
