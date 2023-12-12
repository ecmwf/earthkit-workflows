import xarray as xr

from .array_api import ArrayApiBackend
from .xarray_backend import XArrayBackend

BACKENDS = {}


def register(type, backend):
    assert type not in BACKENDS
    BACKENDS[type] = backend


register(xr.DataArray, XArrayBackend)
register(xr.Dataset, XArrayBackend)


def __getattr__(name: str) -> callable:
    def f(*args, **kwargs):
        array_type = type(args[0])
        # print(array_type, BACKENDS)
        # assert all([array_type == type(args[x]) for x in range(1, len(args))])
        backend = BACKENDS.get(array_type, None)
        if backend is None:
            # Fall back on array api
            backend = ArrayApiBackend
        return getattr(backend, name)(*args, **kwargs)

    f.__name__ = name
    return f
