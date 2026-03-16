# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for cascade exception serialization/deserialization."""

from cascade.low.exceptions import (
    CascadeError,
    CascadeInfrastructureError,
    CascadeInternalError,
    CascadeUserError,
    des,
    ser,
)


def test_ser_des_cascade_internal_error():
    """Test serialization and deserialization of CascadeInternalError."""
    exc = CascadeInternalError("something went wrong")
    serialized = ser(exc)
    deserialized = des(serialized)

    assert isinstance(deserialized, CascadeInternalError)
    assert deserialized.description == "something went wrong"
    assert deserialized.parent is None


def test_ser_des_cascade_infrastructure_error():
    """Test serialization and deserialization of CascadeInfrastructureError."""
    exc = CascadeInfrastructureError("network failure")
    serialized = ser(exc)
    deserialized = des(serialized)

    assert isinstance(deserialized, CascadeInfrastructureError)
    assert deserialized.description == "network failure"
    assert deserialized.parent is None


def test_ser_des_cascade_user_error():
    """Test serialization and deserialization of CascadeUserError."""
    exc = CascadeUserError("invalid configuration")
    serialized = ser(exc)
    deserialized = des(serialized)

    assert isinstance(deserialized, CascadeUserError)
    assert deserialized.description == "invalid configuration"
    assert deserialized.parent is None


def test_ser_des_cascade_error():
    """Test serialization and deserialization of base CascadeError."""
    exc = CascadeError("general error")
    serialized = ser(exc)
    deserialized = des(serialized)

    assert isinstance(deserialized, CascadeError)
    assert deserialized.description == "general error"
    assert deserialized.parent is None


def test_ser_des_with_parent():
    """Test that parent field is not preserved (set to None)."""
    parent_exc = ValueError("parent error")
    exc = CascadeInternalError("child error", parent=parent_exc)
    serialized = ser(exc)
    deserialized = des(serialized)

    assert isinstance(deserialized, CascadeInternalError)
    assert deserialized.description == "child error"
    assert deserialized.parent is None


def test_des_invalid_string():
    """Test deserialization of invalid string returns CascadeInternalError."""
    invalid_str = "not a valid exception format"
    deserialized = des(invalid_str)

    assert isinstance(deserialized, CascadeInternalError)
    assert deserialized.description == invalid_str
    assert deserialized.parent is None


def test_des_unknown_exception_class():
    """Test deserialization of unknown exception class returns CascadeInternalError."""
    invalid_str = "UnknownError('some error')"
    deserialized = des(invalid_str)

    assert isinstance(deserialized, CascadeInternalError)
    assert deserialized.description == invalid_str
    assert deserialized.parent is None


def test_ser_des_roundtrip_all_types():
    """Test roundtrip serialization for all exception types."""
    exceptions = [
        CascadeError("test error"),
        CascadeInternalError("internal error"),
        CascadeInfrastructureError("infra error"),
        CascadeUserError("user error"),
    ]

    for exc in exceptions:
        serialized = ser(exc)
        deserialized = des(serialized)
        assert type(deserialized) == type(exc)
        assert deserialized.description == exc.description
        assert deserialized.parent is None
