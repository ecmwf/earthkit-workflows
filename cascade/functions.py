import xarray as xr
from meteokit import extreme as extreme
import numpy as np
from io import BytesIO

import pyfdb
import mir
import eccodes

from pproc.common.io import read_grib_messages


def efi_xr2np(clim: xr.DataArray, ens: xr.DataArray, eps: float) -> xr.DataArray:
    clim = clim.reindex(
        quantile=sorted(
            clim.coords["quantile"].values, key=lambda x: int(x.split(":")[0])
        )
    )
    ens_values = ens.values
    if ens_values.ndim == 1:
        ens_values = np.reshape(ens.values, (1, -1))
    return xr.DataArray(
        extreme.efi(clim.values, ens_values, eps),
        attrs=ens.attrs,
    )


def sot_xr2np(
    clim: xr.DataArray, ens: xr.DataArray, number: int, eps: float
) -> xr.DataArray:
    clim = clim.reindex(
        quantile=sorted(
            clim.coords["quantile"].values, key=lambda x: int(x.split(":")[0])
        )
    )
    ens_values = ens.values
    if ens_values.ndim == 1:
        ens_values = np.reshape(ens.values, (1, -1))
    return xr.DataArray(
        extreme.sot(clim.values, ens_values, number, eps),
        attrs=ens.attrs,
    )


def mir_wind(
    request: dict, filename: str, interpolation_keys: dict | None = None
) -> xr.DataArray:
    out = BytesIO()
    cached_file = filename.format_map(request)
    inp = mir.MultiDimensionalGribFileInput(cached_file, 2)
    job = mir.Job(vod2uv="1", **interpolation_keys)
    job.execute(inp, out)
    out.seek(0)
    reader = eccodes.StreamReader(out)
    fields_dims = [key for key in request if isinstance(request[key], (list, range))]
    fields = read_grib_messages(reader, fields_dims).to_xarray()
    assert len(fields.coords["param"]) == 2
    wind_speed = xr.apply_ufunc(
        np.linalg.norm,
        fields,
        input_core_dims=[["param"]],
        kwargs={"axis": -1},
        keep_attrs=True,
    )
    return wind_speed
