import sys
import xarray as xr

from .array_api import ArrayApiBackend
from .xarray_backend import XArrayBackend

BACKENDS = {
    xr.DataArray: XArrayBackend,
    xr.Dataset: XArrayBackend,
}


def register(type, backend):
    assert type not in BACKENDS
    BACKENDS[type] = backend


def array_module(*arrays):
    array_type = type(arrays[0])
    assert all([array_type == type(arrays[x]) for x in range(1, len(arrays))])
    backend = BACKENDS.get(array_type, None)
    if backend is not None:
        return backend

    if "jax" in sys.modules:
        import jax
        from .jax_backend import JaxBackend

        if isinstance(arrays[0], jax.Array):
            return JaxBackend

    # Fall back on array api
    return ArrayApiBackend


def __getattr__(name: str) -> callable:
    def f(*args, **kwargs):
        backend = array_module(args[0])
        return getattr(backend, name)(*args, **kwargs)

    f.__name__ = name
    return f
