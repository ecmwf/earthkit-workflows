import xarray as xr
from io import BytesIO
import numpy as np

from pproc.common.io import (
    target_from_location,
    write_grib,
    FileTarget,
    FileSetTarget,
)
from pproc.common import ResourceMeter

import mir
from earthkit.data.sources.stream import StreamSource
from earthkit.data import FieldList
from earthkit.data.sources import Source, from_source
from earthkit.data.sources.numpy_list import NumpyFieldList

from .grib import GribBufferMetaData, basic_headers, buffer_to_template


def mir_job(input: mir.MultiDimensionalGribFileInput, mir_options: dict) -> Source:
    job = mir.Job(**mir_options)
    stream = BytesIO()
    job.execute(input, stream)
    stream.seek(0)
    return StreamSource(stream, batch_size=0).mutate()


def fdb_retrieve(request: dict, *, stream: bool = True) -> Source:
    mir_options = request.pop("interpolate", None)
    # print("REQUEST", request, "STREAM", stream)
    if mir_options:
        reader = from_source("fdb", request, stream=stream)
        if stream:
            ds = mir_job(reader._stream, mir_options)
        else:
            size = len(request["param"]) if isinstance(request["param"], list) else 1
            inp = mir.MultiDimensionalGribFileInput(reader.path, size)
            ds = mir_job(inp, mir_options)
        return ds
    return from_source("fdb", request, batch_size=0, stream=stream)


def mars_retrieve(request: dict) -> Source:
    mir_options = request.pop("interpolate", None)
    ds = from_source("mars", request)
    if mir_options:
        size = len(request["param"]) if isinstance(request["param"], list) else 1
        inp = mir.MultiDimensionalGribFileInput(ds.path, size)
        ds = mir_job(inp, mir_options)
    return ds


def file_retrieve(path: str, request: dict) -> Source:
    mir_options = request.pop("interpolate", None)
    location = path.format_map(request)
    if mir_options:
        size = len(request["param"]) if isinstance(request["param"], list) else 1
        inp = mir.MultiDimensionalGribFileInput(location, size)
        return mir_job(inp, mir_options)
    return from_source("file", location)


def retrieve(source: str, request: dict, **kwargs) -> NumpyFieldList:
    req = request.copy()
    with ResourceMeter(f"retrieve: source {source}, request {request}"):
        if source == "fdb":
            ret_sources = fdb_retrieve(req, **kwargs)
        elif source == "mars":
            ret_sources = mars_retrieve(req)
        elif source == "fileset":
            path = req.pop("location")
            ret_sources = file_retrieve(path, req)
        else:
            raise NotImplementedError("Source {source} not supported.")
    ret = None
    for source in ret_sources:
        grib_metadata = source.metadata()._handle.copy()
        grib_metadata.set_array("values", np.zeros(source.values.shape))
        field_list = FieldList.from_numpy(
            np.asarray([source.values]),
            GribBufferMetaData(grib_metadata.get_buffer()),
        )
        if ret is None:
            ret = field_list
        else:
            ret += field_list
    assert ret is not None, f"No data retrieved from {source} for request {request}"
    return ret


def write(loc: str, data: xr.DataArray, grib_sets: dict):
    target = target_from_location(loc)
    if isinstance(target, (FileTarget, FileSetTarget)):
        # Allows file to be appended on each write call
        target.enable_recovery()
    assert len(data) == 1
    metadata = grib_sets.copy()
    metadata.update(data.metadata()[0]._d)
    metadata = basic_headers(metadata)
    set_missing = [key for key, value in metadata.items() if value == "MISSING"]
    for missing_key in set_missing:
        metadata.pop(missing_key)

    template = buffer_to_template(data.metadata()[0]).override(metadata)

    for missing_key in set_missing:
        template._handle.set_missing(missing_key)
    with ResourceMeter(f"write: target {loc}"):
        write_grib(target, template._handle, data[0].values)
