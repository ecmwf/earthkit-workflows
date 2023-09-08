import xarray as xr

from pproc.common.io import target_from_location, write_grib, fdb_read, fdb


def retrieve(request: dict):
    interpolation_keys = request.pop("interpolate")
    return fdb_read(fdb(), request, interpolation_keys)


def write(loc: str, data: xr.DataArray, grib_sets: dict):
    target = target_from_location(loc)
    template = data.attrs["grib_template"]
    template.set(grib_sets)
    write_grib(target, template, data.data)
