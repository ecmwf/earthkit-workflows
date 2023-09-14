import xarray as xr
from meteokit import extreme as extreme
import numpy as np


def efi_xr2np(clim, ens, eps):
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


def sot_xr2np(clim, ens, number, eps):
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
