# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from earthkit.workflows import mark as ekw_mark
from earthkit.workflows.fluent import Payload, PayloadBuildingContext
from earthkit.workflows.nodetree import nodetree_array

from .helpers import mock_action


def test_payload_metadata():
    """Test payload metadata is passed to the action"""
    action = mock_action((1, 1))

    test_payload = Payload(lambda x: x, metadata={"test_metadata": True})

    mapped_action = action.map(test_payload)

    assert all(
        map(
            lambda x: x.payload.metadata["test_metadata"],
            np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten(),
        )
    )


def test_payload_metadata_with_function():
    """Test payload metadata is passed to the action"""
    action = mock_action((1, 1))

    mult_action = action.multiply(2, payload_metadata={"test_metadata": True})

    assert all(
        map(
            lambda x: x.payload.metadata["test_metadata"],
            np.atleast_1d(nodetree_array(mult_action.nodes).values).flatten(),
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
            np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten(),
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
            np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten(),
        )
    )


# ---------------------------------------------------------------------------
# PayloadBuildingContext tests
# ---------------------------------------------------------------------------


def test_payload_building_context_basic():
    """Metadata from the context is injected into every Payload created inside."""
    with PayloadBuildingContext(env="test"):
        p = Payload(lambda x: x)

    assert p.metadata["env"] == "test"


def test_payload_building_context_not_applied_outside():
    """Metadata is NOT injected into Payloads created outside the context."""
    with PayloadBuildingContext(env="test"):
        pass

    p = Payload(lambda x: x)
    assert "env" not in p.metadata


def test_payload_building_context_nested_merge():
    """Inner context values override outer ones; all keys are present."""
    with PayloadBuildingContext(key1="outer"):
        with PayloadBuildingContext(key2="middle"):
            with PayloadBuildingContext(key1="inner"):
                p = Payload(lambda x: x)

    assert p.metadata["key1"] == "inner"
    assert p.metadata["key2"] == "middle"


def test_payload_building_context_direct_param_wins():
    """Direct metadata= argument overrides context-provided metadata."""
    with PayloadBuildingContext(key1="from_context", key2="from_context"):
        p = Payload(lambda x: x, metadata={"key1": "direct", "key3": "direct"})

    assert p.metadata["key1"] == "direct"
    assert p.metadata["key2"] == "from_context"
    assert p.metadata["key3"] == "direct"


def test_payload_building_context_full_example():
    """Reproduces the docstring example with all three sources combined."""
    with PayloadBuildingContext(key1="value1"):
        with PayloadBuildingContext(key2="value2"):
            with PayloadBuildingContext(key1="value3"):
                p = Payload(lambda x: x, metadata={"key3": "value4"})

    assert p.metadata["key1"] == "value3"
    assert p.metadata["key2"] == "value2"
    assert p.metadata["key3"] == "value4"


def test_payload_building_context_on_action_map():
    """Context metadata propagates to nodes created via Action.map."""
    action = mock_action((2, 2))

    with PayloadBuildingContext(stage="production"):
        mapped = action.map(lambda x: x)

    nodes = np.atleast_1d(nodetree_array(mapped.nodes).values).flatten()
    assert all(n.payload.metadata["stage"] == "production" for n in nodes)


def test_payload_building_context_does_not_bleed_between_sibling_contexts():
    """Sibling contexts do not interfere with each other."""
    with PayloadBuildingContext(key="first"):
        p1 = Payload(lambda x: x)

    with PayloadBuildingContext(key="second"):
        p2 = Payload(lambda x: x)

    assert p1.metadata["key"] == "first"
    assert p2.metadata["key"] == "second"
