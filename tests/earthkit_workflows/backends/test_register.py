# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from earthkit.workflows import backends
from earthkit.workflows.backends.arrayapi import ArrayAPIBackend


class CustomStrBackend:
    pass


class CustomIntBackend:
    pass


class Special(str):
    pass


@pytest.fixture
def empty_backend():
    """Fixture to reset the BACKENDS dictionary before each test."""
    BACKEND_COPY = backends.BACKENDS.copy()
    backends.BACKENDS.clear()
    backends.BACKENDS.update(
        {
            object: ArrayAPIBackend,
        }
    )
    yield
    backends.BACKENDS.clear()
    backends.BACKENDS.update(BACKEND_COPY)


def test_single_register(empty_backend):
    backends.register(str, CustomStrBackend)
    backends.register(int, CustomStrBackend)

    assert backends.array_module("test") == CustomStrBackend
    assert backends.array_module(0) == CustomStrBackend
    assert backends.array_module(0.0) == ArrayAPIBackend


def test_nested_register(empty_backend):
    backends.register(str, CustomStrBackend)
    backends.register(Special, CustomIntBackend)

    assert backends.array_module("test") == CustomStrBackend
    assert backends.array_module(Special("test")) == CustomIntBackend
    assert backends.array_module(0) == ArrayAPIBackend


def test_lookup_register(empty_backend):
    backends.register(str, CustomStrBackend)

    assert backends.array_module("test") == CustomStrBackend
    assert backends.array_module(Special("test")) == CustomStrBackend
