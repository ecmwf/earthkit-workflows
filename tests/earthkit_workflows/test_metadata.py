# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pytest

from earthkit.workflows import mark as ekw_mark
from earthkit.workflows.context import NodeMetadataContext
from earthkit.workflows.fluent import create_task_instance, from_source
from earthkit.workflows.metadata import Artifacts, BuilderMetadata, NodeMetadata, Requirements

SOURCE_ACTION = from_source("test", datacubes={"dim": [0], "dim1": [0]})


def test_node_metadata():
    """Test node metadata is passed to the node and task definition"""
    task = create_task_instance(
        lambda x: x,
        requirements=Requirements(needs_gpu=True, environment=["test"]),
    )
    mapped_action = SOURCE_ACTION.map(task)

    assert all(
        x.metadata.requirements.needs_gpu
        and x.metadata.requirements.environment == ["test"]
        and x.payload.definition.needs_gpu
        and x.metadata.requirements.environment == x.payload.definition.environment
        for x in mapped_action.nodes.values()
    )


def test_node_metadata_with_function():
    """Test node metadata is passed in the action"""
    mapped_action = SOURCE_ACTION.map(
        lambda x: x,
        node_metadata=NodeMetadata(
            requirements=Requirements(needs_gpu=True, environment=["test"]),
            builder=BuilderMetadata(blockId="test_block"),
            artifacts=Artifacts(artifact_urls={"test_artifact": "http://example.com/artifact"}),
        ),
    )

    assert all(
        x.metadata.requirements.needs_gpu
        and x.metadata.requirements.environment == ["test"]
        and x.payload.definition.needs_gpu
        and x.metadata.requirements.environment == x.payload.definition.environment
        for x in mapped_action.nodes.values()
    )
    assert all(x.metadata.artifacts.artifact_urls == {"test_artifact": "http://example.com/artifact"} for x in mapped_action.nodes.values())
    assert all(x.metadata.builder.blockId == "test_block" for x in mapped_action.nodes.values())


def test_payload_metadata_from_marks_generic():
    """Test payload metadata from generic mark"""

    @ekw_mark.add_execution_metadata(needs_gpu=True)
    def test_function(x):
        return x

    mapped_action = SOURCE_ACTION.map(test_function)

    assert all(
        map(
            lambda x: x.payload.definition.needs_gpu,
            mapped_action.nodes.values(),
        )
    )


def test_payload_metadata_from_marks_explicit():
    @ekw_mark.needs_gpu
    def test_function(x):
        return x

    mapped_action = SOURCE_ACTION.map(test_function)

    assert all(
        map(
            lambda x: x.payload.definition.needs_gpu,
            mapped_action.nodes.values(),
        )
    )


# ---------------------------------------------------------------------------
# NodeMetadataContext tests
# ---------------------------------------------------------------------------


def test_node_building_context_basic():
    """Metadata from the context is injected into every Payload/Node created inside."""
    with NodeMetadataContext(requirements=Requirements(environment=["test"]), builder=BuilderMetadata(blockId="test_block")):
        mapped_action = SOURCE_ACTION.map(lambda x: x)
    assert all(
        map(
            lambda x: (
                x.metadata.builder.blockId == "test_block"
                and not x.payload.definition.needs_gpu
                and x.payload.definition.environment == ["test"]
            ),
            mapped_action.nodes.values(),
        )
    )


def test_node_building_context_not_applied_outside():
    """Metadata is NOT injected into Payloads/Nodes created outside the context."""
    with NodeMetadataContext(requirements=Requirements(environment=["test"])):
        pass

    mapped_action = SOURCE_ACTION.map(lambda x: x)
    assert all(
        map(
            lambda x: x.metadata.builder.blockId is None and not x.payload.definition.needs_gpu and x.payload.definition.environment == [],
            mapped_action.nodes.values(),
        )
    )


def test_node_building_context_nested_merge():
    """Inner context values override outer ones; all keys are present."""
    with NodeMetadataContext(requirements=Requirements(needs_gpu=False, environment=["outer"])):
        with NodeMetadataContext(requirements=Requirements(environment=["middle"]), builder=BuilderMetadata(blockId="test_block")):
            with NodeMetadataContext(requirements=Requirements(needs_gpu=True)):
                mapped_action = SOURCE_ACTION.map(lambda x: x)

    nodes = mapped_action.nodes.values()
    assert all(n.payload.definition.needs_gpu and n.metadata.requirements.needs_gpu for n in nodes)
    assert all(set(n.payload.definition.environment) == {"middle", "outer"} for n in nodes)
    assert all(n.metadata.builder.blockId == "test_block" for n in nodes)


def test_node_building_context_direct_param_wins():
    """Direct metadata= argument overrides context-provided metadata."""
    with NodeMetadataContext(requirements=Requirements(needs_gpu=False, environment=["from_context"])):

        @ekw_mark.add_execution_metadata(needs_gpu=True, environment=["direct"])
        def test_function(x):
            return x

        mapped_action = SOURCE_ACTION.map(test_function)

    nodes = mapped_action.nodes.values()
    assert all(n.payload.definition.needs_gpu for n in nodes)
    assert all(set(n.payload.definition.environment) == {"direct", "from_context"} for n in nodes)


@pytest.mark.parametrize(
    "func", [lambda x: x, "test_func", create_task_instance("test_func")], ids=["callable", "entrypoint", "task_instance"]
)
def test_node_building_context_full_example(func):
    """Reproduces the docstring example with all three sources combined."""
    with NodeMetadataContext(requirements=Requirements(needs_gpu=False), builder=BuilderMetadata(blockId="test_block")):
        with NodeMetadataContext(requirements=Requirements(environment=[]), artifacts=Artifacts(artifact_urls={"test_artifact": "url_1"})):
            with NodeMetadataContext(requirements=Requirements(needs_gpu=True), builder=BuilderMetadata(blockId="inner_block")):
                mapped_action = SOURCE_ACTION.map(
                    func,
                    node_metadata=NodeMetadata(
                        requirements=Requirements(environment=["value4"]),
                        artifacts=Artifacts(artifact_urls={"test_artifact": "url_1"}),
                    ),
                )

    nodes = mapped_action.nodes.values()
    assert all(n.payload.definition.needs_gpu and n.metadata.requirements.needs_gpu for n in nodes)
    assert all(set(n.payload.definition.environment) == {"value4"} for n in nodes)
    assert all(n.metadata.builder.blockId == "inner_block" for n in nodes)
    assert all(n.metadata.artifacts.artifact_urls == {"test_artifact": "url_1"} for n in nodes)


def test_node_building_context_does_not_bleed_between_sibling_contexts():
    """Sibling contexts do not interfere with each other."""
    with NodeMetadataContext(requirements=Requirements(environment=["first"])):
        a1 = SOURCE_ACTION.map(lambda x: x)

    with NodeMetadataContext(requirements=Requirements(environment=["second"])):
        a2 = SOURCE_ACTION.map(lambda x: x)

    assert all(set(n.payload.definition.environment) == {"first"} for n in a1.nodes.values())
    assert all(set(n.payload.definition.environment) == {"second"} for n in a2.nodes.values())
