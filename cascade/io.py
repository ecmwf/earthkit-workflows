import xarray as xr
from io import BytesIO
import numpy as np

from pproc.common.io import (
    target_from_location,
    write_grib,
    FileTarget,
    FileSetTarget,
)
import pyfdb
import mir
import earthkit.data
from earthkit.data.sources.stream import StreamSource


def mir_job(input, mir_options):
    job = mir.Job(**mir_options)
    stream = BytesIO()
    job.execute(input, stream)
    stream.seek(0)
    return StreamSource(stream, batch_size=0).mutate()


def fdb_retrieve(request: dict, stream=True):
    mir_options = request.pop("interpolate", None)
    if mir_options:
        reader = earthkit.data.from_source("fdb", request, stream=stream)
        if stream:
            ds = mir_job(reader._stream, mir_options)
        else:
            size = len(request["param"]) if isinstance(request["param"], list) else 1
            inp = mir.MultiDimensionalGribFileInput(reader.path, size)
            ds = mir_job(inp, mir_options)
        return ds
    return earthkit.data.from_source("fdb", request, batch_size=0, stream=stream)


def mars_retrieve(request: dict):
    mir_options = request.pop("interpolate", None)
    ds = earthkit.data.from_source("mars", request)
    if mir_options:
        size = len(request["param"]) if isinstance(request["param"], list) else 1
        inp = mir.MultiDimensionalGribFileInput(ds.path, size)
        ds = mir_job(inp, mir_options)
    return ds


def file_retrieve(path: str, request):
    mir_options = request.pop("interpolate", None)
    location = path.format_map(request)
    if mir_options:
        size = len(request["param"]) if isinstance(request["param"], list) else 1
        inp = mir.MultiDimensionalGribFileInput(location, size)
        return mir_job(inp, mir_options)
    return earthkit.data.from_source("file", location)


def retrieve(source: str, request: dict, **kwargs):
    req = request.copy()
    if source == "fdb":
        return fdb_retrieve(req, kwargs)
    if source == "mars":
        return mars_retrieve(req)
    if source == "fileset":
        path = req.pop("location")
        return file_retrieve(path, req)
    raise NotImplementedError("Source {source} not supported.")


def write(loc: str, data: xr.DataArray, grib_sets: dict):
    target = target_from_location(loc)
    if isinstance(target, (FileTarget, FileSetTarget)):
        # Allows file to be appended on each write call
        target.enable_recovery()
    template = data.attrs["grib_template"].copy()
    template.set(grib_sets)
    write_grib(target, template, data.data)
