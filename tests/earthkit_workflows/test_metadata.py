# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from earthkit.workflows import mark as ekw_mark
from earthkit.workflows.fluent import PayloadBuildingContext, create_task_instance
from earthkit.workflows.nodetree import nodetree_array

from .helpers import mock_action


def test_payload_metadata():
    """Test payload metadata is passed to the action"""
    action = mock_action((1, 1))

    test_payload = create_task_instance(lambda x: x, payload_metadata={"needs_gpu": True})

    mapped_action = action.map(test_payload)

    assert all(
        map(
            lambda x: x.payload.definition.needs_gpu,
            np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten(),
        )
    )


def test_payload_metadata_with_function():
    """Test payload metadata is passed to the action"""
    action = mock_action((1, 1))

    mult_action = action.multiply(2, payload_metadata={"needs_gpu": True, "environment": ["test"]})

    assert all(
        map(
            lambda x: x.payload.definition.needs_gpu and x.payload.definition.environment == ["test"],
            np.atleast_1d(nodetree_array(mult_action.nodes).values).flatten(),
        )
    )


def test_payload_metadata_from_marks_generic():
    """Test payload metadata from generic mark"""
    action = mock_action((1, 1))

    @ekw_mark.add_execution_metadata(needs_gpu=True)
    def test_function(x):
        return x

    mapped_action = action.map(test_function)

    assert all(
        map(
            lambda x: x.payload.definition.needs_gpu,
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
            lambda x: x.payload.definition.needs_gpu,
            np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten(),
        )
    )


# ---------------------------------------------------------------------------
# PayloadBuildingContext tests
# ---------------------------------------------------------------------------


def test_payload_building_context_basic():
    """Metadata from the context is injected into every Payload created inside."""
    with PayloadBuildingContext(environment=["test"]):
        p = create_task_instance(lambda x: x)
    assert p.definition.environment == ["test"]


def test_payload_building_context_not_applied_outside():
    """Metadata is NOT injected into Payloads created outside the context."""
    with PayloadBuildingContext(environment=["test"]):
        pass

    p = create_task_instance(lambda x: x)
    assert "environment" not in p.definition


def test_payload_building_context_nested_merge():
    """Inner context values override outer ones; all keys are present."""
    with PayloadBuildingContext(needs_gpu=False, environment=["outer"]):
        with PayloadBuildingContext(environment=["middle"]):
            with PayloadBuildingContext(needs_gpu=True):
                p = create_task_instance(lambda x: x)

    assert p.definition.needs_gpu is True
    assert set(p.definition.environment) == {"middle", "outer"}


def test_payload_building_context_direct_param_wins():
    """Direct metadata= argument overrides context-provided metadata."""
    with PayloadBuildingContext(needs_gpu=False, environment=["from_context"]):
        p = create_task_instance(lambda x: x, payload_metadata={"needs_gpu": True, "environment": ["direct"]})

    assert p.definition.needs_gpu
    assert set(p.definition.environment) == {"direct", "from_context"}


def test_payload_building_context_full_example():
    """Reproduces the docstring example with all three sources combined."""
    with PayloadBuildingContext(needs_gpu=False):
        with PayloadBuildingContext(environment=[]):
            with PayloadBuildingContext(needs_gpu=True):
                p = create_task_instance(lambda x: x, payload_metadata={"environment": ["value4"]})

    assert p.definition.needs_gpu is True
    assert p.definition.environment == ["value4"]


def test_payload_building_context_on_action_map():
    """Context metadata propagates to nodes created via Action.map."""
    action = mock_action((2, 2))

    with PayloadBuildingContext(needs_gpu=True):
        mapped = action.map(lambda x: x)

    nodes = np.atleast_1d(nodetree_array(mapped.nodes).values).flatten()
    assert all(n.payload.definition.needs_gpu for n in nodes)


def test_payload_building_context_does_not_bleed_between_sibling_contexts():
    """Sibling contexts do not interfere with each other."""
    with PayloadBuildingContext(environment=["first"]):
        p1 = create_task_instance(lambda x: x)

    with PayloadBuildingContext(environment=["second"]):
        p2 = create_task_instance(lambda x: x)

    assert p1.definition.environment == ["first"]
    assert p2.definition.environment == ["second"]
