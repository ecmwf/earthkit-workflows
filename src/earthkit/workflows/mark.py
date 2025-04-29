# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def add_execution_metadata(**kwargs) -> Callable[[F], F]:
    """Add execution metadata to a function."""

    def decorator(func: F) -> F:
        if not hasattr(func, "_cascade"):
            func._cascade = {}

        func._cascade.update(kwargs)
        return func

    return decorator


def needs_gpu(func: F) -> F:
    """Decorator to mark a function as needing a GPU."""
    return add_execution_metadata(needs_gpu=True)(func)


def needs_cpu(func: F) -> F:
    """Decorator to mark a function as needing a CPU."""
    return add_execution_metadata(needs_cpu=True)(func)


def environment_requirements(env: list[str]) -> Callable[[F], F]:
    """Decorator to mark a function as needing a specific environment.

    Environment is a list of strings of packages, e.g. ["numpy<2.0.0"].
    """
    return add_execution_metadata(environment=env)
