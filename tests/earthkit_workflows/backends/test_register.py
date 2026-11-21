# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator

import pytest

from earthkit.workflows import backends
from earthkit.workflows.backends.arrayapi import ArrayAPIBackend


class CustomStrBackend:
    pass


class CustomIntBackend:
    pass


class Special(str):
    pass


class Mixin:
    pass


class Registered:
    pass


class Derived(Mixin, Registered):
    """Registered ancestor is not on the first-base chain."""


class Unrelated(Mixin):
    pass


@pytest.fixture
def empty_backend() -> Iterator[None]:
    """Fixture to reset the BACKENDS dictionary before each test."""
    BACKEND_COPY = backends.ARRAY_BACKENDS.copy()
    backends.ARRAY_BACKENDS.clear()
    yield
    backends.ARRAY_BACKENDS.clear()
    backends.ARRAY_BACKENDS.update(BACKEND_COPY)


@pytest.fixture
def restore_backends() -> Iterator[None]:
    """Restore ARRAY_BACKENDS after the test without clearing it beforehand."""
    BACKEND_COPY = backends.ARRAY_BACKENDS.copy()
    yield
    backends.ARRAY_BACKENDS.clear()
    backends.ARRAY_BACKENDS.update(BACKEND_COPY)


def test_single_register(empty_backend):
    backends.register("builtins.str", CustomStrBackend)
    backends.register("builtins.int", CustomIntBackend)
    backends.register("builtins.float", ArrayAPIBackend)

    assert backends.array_module("test") == CustomStrBackend
    assert backends.array_module(0) == CustomIntBackend
    assert backends.array_module(0.0) == ArrayAPIBackend


def test_nested_register(empty_backend):
    backends.register("builtins.str", CustomStrBackend)
    backends.register("builtins.int", ArrayAPIBackend)
    backends.register("test_register.Special", CustomIntBackend)

    assert backends.array_module("test") == CustomStrBackend
    assert backends.array_module(Special("test")) == CustomIntBackend
    assert backends.array_module(0) == ArrayAPIBackend


def test_mro_register(restore_backends):
    backends.register(f"{Registered.__module__}.{Registered.__name__}", CustomStrBackend)

    assert Derived.__bases__[0] is Mixin
    assert backends.array_module(Derived()) == CustomStrBackend
    assert backends.array_module(Registered()) == CustomStrBackend


def test_mro_no_registered_ancestor(restore_backends):
    backends.register(f"{Registered.__module__}.{Registered.__name__}", CustomStrBackend)

    with pytest.raises(TypeError, match="No backend registered"):
        backends.array_module(Unrelated())
