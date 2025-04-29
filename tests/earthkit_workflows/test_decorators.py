# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from earthkit.workflows.decorators import as_payload
from earthkit.workflows.fluent import Payload


def test_as_payload():
    """Test the as_payload decorator"""

    @as_payload
    def test_function(x, y):
        return x + y

    payload = test_function(1, 2, metadata={"test_metadata": True})

    assert isinstance(payload, Payload)
    assert payload.metadata["test_metadata"]
    assert payload.args == [1, 2]
    assert payload.kwargs == {}
