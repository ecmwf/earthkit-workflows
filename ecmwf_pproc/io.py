from io import BytesIO
import numpy as np
import os
import importlib
from os.path import join as pjoin
import shutil
import sys

import mir
from pproc.common.io import (
    target_from_location,
    write_grib,
    FileTarget,
    FileSetTarget,
)
from pproc.common import ResourceMeter
from pproc.clustereps.__main__ import write_cluster_attr_grib
from pproc.clustereps.cluster import get_output_keys
from pproc.common.io import split_location
from earthkit.data import FieldList
from earthkit.data.sources.stream import StreamSource
from earthkit.data.sources.file import FileSource
from earthkit.data.sources import Source, from_source
from earthkit.data.sources.numpy_list import NumpyFieldList

from .grib import basic_headers
from .wrappers.metadata import GribBufferMetaData


def _source_from_location(loc, sources) -> tuple[str, list[dict]]:
    type_, ident = split_location(loc, default="file")
    requests = sources.get(type_, {}).get(ident, None)
    assert (
        requests is not None
    ), f"Not requests listed for location {loc} in sources {sources}"
    if isinstance(requests, dict):
        requests = [requests]
    return type_, requests


def mir_job(
    input: mir.MultiDimensionalGribFileInput, mir_options: dict, cache: str = None
) -> Source:
    job = mir.Job(**mir_options)
    stream = BytesIO()
    job.execute(input, stream)
    stream.seek(0)
    if cache is None:
        return StreamSource(stream, batch_size=0).mutate()

    with open(cache, "wb") as o, stream as i:
        shutil.copyfileobj(i, o)
    return FileSource(cache).mutate()


def fdb_retrieve(request: dict, *, stream: bool = True) -> Source:
    mir_options = request.pop("interpolate", None)
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
    cache = request.pop("cache", None)
    cache_path = None if cache is None else cache.format_map(request)
    ds = from_source("mars", request)
    if mir_options:
        size = len(request["param"]) if isinstance(request["param"], list) else 1
        inp = mir.MultiDimensionalGribFileInput(ds.path, size)
        ds = mir_job(inp, mir_options, cache_path)
    return ds


def _transform_steps(steps, step_type: type = str):
    if isinstance(steps, (int, str)):
        steps = [steps]
    return list(map(step_type, steps))


def _transform_request(request: dict, step_type: type = str):
    try:
        paramId = int(request["param"])
        del request["param"]
        request["paramId"] = paramId
    except:
        pass
    request["date"] = int(request["date"])
    time = int(request["time"])
    request["time"] = time if time % 100 == 0 else time * 100
    request["step"] = _transform_steps(request["step"], step_type)
    return request


def file_retrieve(path: str, request: dict) -> Source:
    mir_options = request.pop("interpolate", None)
    if mir_options:
        raise NotImplementedError()
    location = path.format_map(request)
    file_ds = from_source("file", location)
    ds = file_ds.sel(_transform_request(request))
    if len(ds) == 0:
        try:
            request["step"] = _transform_steps(request["step"], int)
            ds = file_ds.sel(request)
        except ValueError:
            pass
    return ds


def retrieve(request: dict | list[dict], **kwargs):
    if isinstance(request, dict):
        return retrieve_single_source(request, **kwargs)
    return retrieve_multi_sources(request, **kwargs)


def retrieve_multi_sources(requests: list[dict], **kwargs) -> NumpyFieldList:
    ret = None
    for req in requests:
        try:
            ret = retrieve_single_source(req, **kwargs)
            break
        except AssertionError:
            continue
    assert ret is not None, f"No data retrieved from requests: {requests}"
    return ret


def retrieve_single_source(request: dict, **kwargs) -> NumpyFieldList:
    xp = importlib.import_module(os.getenv("CASCADE_ARRAY_MODULE", "numpy"))

    req = request.copy()
    source = req.pop("source")
    with ResourceMeter(f"RETRIEVE source {source}, request {request}"):
        if source == "fdb":
            ret_sources = fdb_retrieve(req, **kwargs)
        elif source == "mars":
            ret_sources = mars_retrieve(req)
        elif source == "fileset":
            path = req.pop("location")
            ret_sources = file_retrieve(path, req)
        else:
            raise NotImplementedError("Source {source} not supported.")
    assert (
        len(ret_sources) > 0
    ), f"No data retrieved from {source} for request {request}"
    ret = FieldList.from_numpy(
        xp.asarray(ret_sources.values),
        list(map(GribBufferMetaData, ret_sources.metadata())),
    )
    return ret


def write(loc: str, data: NumpyFieldList, grib_sets: dict):
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

    template = data.metadata()[0].buffer_to_metadata().override(metadata)

    for missing_key in set_missing:
        template._handle.set_missing(missing_key)
    with ResourceMeter(
        f"WRITE target {loc}, size {sys.getsizeof(data[0].values)} bytes"
    ):
        write_grib(target, template._handle, data[0].values)


def cluster_write(
    config,
    scenario,
    attribution_output,
    cluster_dests,
):
    metadata, scdata, anom, cluster_att, min_dist = attribution_output
    grib_template = metadata.buffer_to_metadata()
    cluster_type, ind_cl, rep_members, det_index = [
        metadata._d[x] for x in ["type", "ind_cl", "rep_members", "det_index"]
    ]

    keys, steps = get_output_keys(config, grib_template)
    with ResourceMeter(f"Write {scenario} output"):
        ## Write anomalies and cluster scenarios
        dest, adest = cluster_dests
        target = target_from_location(dest)
        anom_target = target_from_location(adest)
        keys["type"] = cluster_type
        write_cluster_attr_grib(
            steps,
            ind_cl,
            rep_members,
            det_index,
            scdata,
            anom,
            cluster_att,
            target,
            anom_target,
            keys,
            ncl_dummy=config.ncl_dummy,
        )

        ## Write report output
        # table: attribution cluster index for all fc clusters, step
        np.savetxt(
            pjoin(
                config.output_root,
                f"{config.step_start}_{config.step_end}dist_index_{scenario}.txt",
            ),
            min_dist,
            fmt="%-10.5f",
            delimiter=3 * " ",
        )

        # table: distance measure for all fc clusters, step
        np.savetxt(
            pjoin(
                config.output_root,
                f"{config.step_start}_{config.step_end}att_index_{scenario}.txt",
            ),
            cluster_att,
            fmt="%-3d",
            delimiter=3 * " ",
        )
