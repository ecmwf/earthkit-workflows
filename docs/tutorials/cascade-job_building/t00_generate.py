"""Generate dummy data -- sorta like meteo data but small and easy to generate locally"""

import numpy as np
import xarray as xr


def generate_data() -> xr.Dataset:
    coords = {
        "x": np.linspace(0, 10, 10),
        "y": np.linspace(0, 10, 10),
        "t": np.linspace(0, 24, 24),
    }
    variables = {
        "precip": np.random.rand(10, 10, 24),
        "temper": np.random.rand(10, 10, 24),
    }
    return xr.Dataset(
        {
            key: xr.DataArray(value, dims=tuple(coords.keys()), coords=coords)
            for key, value in variables.items()
        }
    )
