# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from functools import wraps
from typing import Any, Callable, Concatenate, ParamSpec, ParamSpecArgs, TypeVar

from .fluent import Payload

P = ParamSpec("P")
R = TypeVar("R")


def as_payload(func: Callable[Concatenate[ParamSpecArgs, P], R]):
    """Wrap a function and return a Payload object.

    Forces the function to be called with keyword arguments only, with args being passed
    once the payload is executed from earlier Nodes.

    Set `metadata` to pass metadata to the payload.

    Examples
    --------
        ```python
        @as_payload
        def my_function(a, b, *, keyword):
            pass

        my_function(1, 2, keyword='test')  # Raises an error
        my_function(b=2, keyword='test')  # OK, a will be passed from earlier nodes
        my_function(keyword='test')  # OK, a and b will be passed from earlier nodes

        ```
    """

    @wraps(func, assigned=["__name__", "__doc__"])
    def decorator(*, metadata: dict[str, Any] | None = None, **kwargs) -> Payload:
        return Payload(func, args=None, kwargs=kwargs, metadata=metadata)

    return decorator
