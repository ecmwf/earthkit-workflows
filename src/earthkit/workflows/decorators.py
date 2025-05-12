# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from .fluent import Payload

P = ParamSpec("P")
R = TypeVar("R")


def as_payload(func: Callable[P, R]) -> Callable[P, Payload]:
    """Wrap a function and return a payload object.

    Will pop metadata from kwargs and pass it to the payload.
    """
    from .fluent import Payload

    @wraps(func)
    def decorator(*args, **kwargs) -> Payload:
        metadata = kwargs.pop("metadata", None)
        return Payload(func, args, kwargs, metadata=metadata)

    return decorator
