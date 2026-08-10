# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from earthkit.workflows.decorators import as_payload
from earthkit.workflows.fluent import Payload


@as_payload
def mock_payload_function(x, y, *, keyword):
    return x + y


def test_as_payload():
    """Test the `as_payload` decorator"""
    payload = mock_payload_function(metadata={"needs_gpu": True}, keyword="test")

    assert isinstance(payload, Payload)
    assert payload.definition.needs_gpu
    assert payload.static_input_ps == {}
    assert payload.static_input_kw == {"keyword": "test"}


def test_as_payload_with_args():
    """Test that calling the function with positional arguments raises an error."""
    with pytest.raises(TypeError):
        mock_payload_function(1, 2, metadata={"needs_gpu": True})
