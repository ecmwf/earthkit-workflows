# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Any

import array_api_compat

from .arrayapi import ArrayAPIBackend
from .base import Backend
from .dictionary import DictBackend
from .earthkit import FieldListBackend
from .xarray import XArrayBackend

logger = logging.getLogger(__name__)


ARRAY_BACKENDS = {
    "builtins.dict": DictBackend,
    "builtins.int": ArrayAPIBackend,
    "builtins.float": ArrayAPIBackend,
    "xarray.core.dataarray.DataArray": XArrayBackend,
    "xarray.core.dataset.Dataset": XArrayBackend,
    "earthkit.data.core.fieldlist.FieldList": FieldListBackend,
}

BackendType = type[Backend]


def register(name: str, backend: Any):
    """
    Register a new backend. Backend is matched based on the module name and class name of the array
    type. For example, to register a backend for numpy arrays, you would use "numpy.ndarray"
    as the name.

    Parameters
    ----------
    name : str
        The name of the backend.
    backend : Any
        The backend class to register. Can be subclass of Backend, or custom type
    """
    if not issubclass(type(backend), Backend):
        logger.warning(f"Backend {backend} does not implement the Backend interface. It may not work as expected.")
    if name in ARRAY_BACKENDS:
        logger.warning(f"Overwriting backend for {name}. Existing backend {ARRAY_BACKENDS[name]}.")
    ARRAY_BACKENDS[name] = backend


def array_module(*arrays) -> BackendType:
    """Return the backend module for the given arrays."""
    if not arrays:
        raise ValueError("No arrays provided to determine backend.")
    if array_api_compat.is_array_api_obj(arrays[0]):
        return ArrayAPIBackend
    array_type = type(arrays[0])
    type_id = ".".join([array_type.__module__, array_type.__name__])
    # Sort keys in reverse so more recently added backends take precedence over older ones.
    # This allows for more specific backends to override more general ones.
    sorted_backends = sorted(ARRAY_BACKENDS.items(), key=lambda x: x[0], reverse=True)
    # Checks all bases of the first array type for a registered backend.
    # If no backend is found, after traversing the hierarchy of types
    # a TypeError is raised
    while True:
        for name, backend in sorted_backends:
            if type_id in name:
                logger.debug(f"Using backend {backend} for {type_id}")
                return backend
        bases = array_type.__bases__
        if len(bases) == 0:
            break
        array_type = bases[0]
        type_id = ".".join([array_type.__module__, array_type.__name__])
    raise TypeError(f"No backend registered for array type {type(arrays[0])}. Please register a backend using `register(name, backend)`.")


def method(name: str, *args, **kwargs) -> Any:
    backend = array_module(*args)
    return getattr(backend, name)(*args, **kwargs)


def batchable(name: str) -> bool:
    return name in ["max", "min", "sum", "prod", "var", "concat"]
