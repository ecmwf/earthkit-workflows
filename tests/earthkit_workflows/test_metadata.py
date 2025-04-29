# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from earthkit.workflows import mark as ekw_mark
from earthkit.workflows.fluent import Payload

from .helpers import mock_action


def test_payload_metadata():
    """Test payload metadata is passed to the action"""
    action = mock_action((1, 1))

    test_payload = Payload(lambda x: x, metadata={"test_metadata": True})

    mapped_action = action.map(test_payload)

    assert all(
        map(
            lambda x: x.payload.metadata["test_metadata"],
            np.atleast_1d(mapped_action.nodes.values).flatten(),
        )
    )


def test_payload_metadata_with_function():
    """Test payload metadata is passed to the action"""
    action = mock_action((1, 1))

    mult_action = action.multiply(2, payload_metadata={"test_metadata": True})

    assert all(
        map(
            lambda x: x.payload.metadata["test_metadata"],
            np.atleast_1d(mult_action.nodes.values).flatten(),
        )
    )


def test_payload_metadata_from_marks_generic():
    """Test payload metadata from generic mark"""
    action = mock_action((1, 1))

    @ekw_mark.add_execution_metadata(test_metadata=True)
    def test_function(x):
        return x

    mapped_action = action.map(test_function)

    assert all(
        map(
            lambda x: x.payload.metadata["test_metadata"],
            np.atleast_1d(mapped_action.nodes.values).flatten(),
        )
    )


def test_payload_metadata_from_marks_explicit():
    action = mock_action((1, 1))

    @ekw_mark.needs_gpu
    def test_function(x):
        return x

    mapped_action = action.map(test_function)

    assert all(
        map(
            lambda x: x.payload.metadata["needs_gpu"],
            np.atleast_1d(mapped_action.nodes.values).flatten(),
        )
    )
