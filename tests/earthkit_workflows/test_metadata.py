# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np

from earthkit.workflows import mark as ekw_mark
from earthkit.workflows.fluent import NodeMetadataContext, create_task_instance
from earthkit.workflows.graph import nodes
from earthkit.workflows.metadata import Artifacts, BuilderMetadata, NodeMetadata, Requirements
from earthkit.workflows.nodetree import nodetree_array

from .helpers import mock_action


def test_node_metadata():
    """Test node metadata is passed to the node and task definition"""
    action = mock_action((1, 1))

    task = create_task_instance(
        lambda x: x,
        requirements=Requirements(needs_gpu=True, environment=["test"]),
        artifacts=Artifacts(artifact_urls={"test_artifact": "http://example.com/artifact"}),
    )
    mapped_action = action.map(task)

    nodes = np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten()
    assert all(
        x.metadata.requirements.needs_gpu
        and x.metadata.requirements.environment == ["test"]
        and x.payload.definition.needs_gpu
        and x.metadata.requirements.environment == x.payload.definition.environment
        for x in nodes
    )
    assert all(
        x.metadata.artifacts.artifact_urls == {"test_artifact": "http://example.com/artifact"}
        and x.metadata.artifacts.artifact_urls == x.payload.artifact_urls
        for x in nodes
    )


def test_node_metadata_with_function():
    """Test node metadata is passed in the action"""
    action = mock_action((1, 1))

    mapped_action = action.map(
        lambda x: x,
        node_metadata=NodeMetadata(
            requirements=Requirements(needs_gpu=True, environment=["test"]),
            builder=BuilderMetadata(blockId="test_block"),
            artifacts=Artifacts(artifact_urls={"test_artifact": "http://example.com/artifact"}),
        ),
    )

    nodes = np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten()
    assert all(
        x.metadata.requirements.needs_gpu
        and x.metadata.requirements.environment == ["test"]
        and x.payload.definition.needs_gpu
        and x.metadata.requirements.environment == x.payload.definition.environment
        for x in nodes
    )
    assert all(
        x.metadata.artifacts.artifact_urls == {"test_artifact": "http://example.com/artifact"}
        and x.metadata.artifacts.artifact_urls == x.payload.artifact_urls
        for x in nodes
    )
    assert all(x.metadata.builder.blockId == "test_block" for x in nodes)


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
# NodeMetadataContext tests
# ---------------------------------------------------------------------------


def test_node_building_context_basic():
    """Metadata from the context is injected into every Payload/Node created inside."""
    action = mock_action((1, 1))
    with NodeMetadataContext(requirements=Requirements(environment=["test"]), builder=BuilderMetadata(blockId="test_block")):
        mapped_action = action.map(lambda x: x)
    assert all(
        map(
            lambda x: (
                x.metadata.builder.blockId == "test_block"
                and not x.payload.definition.needs_gpu
                and x.payload.definition.environment == ["test"]
            ),
            np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten(),
        )
    )


def test_node_building_context_not_applied_outside():
    """Metadata is NOT injected into Payloads/Nodes created outside the context."""
    action = mock_action((1, 1))
    with NodeMetadataContext(requirements=Requirements(environment=["test"])):
        pass

    mapped_action = action.map(lambda x: x)
    assert all(
        map(
            lambda x: x.metadata.builder.blockId is None and not x.payload.definition.needs_gpu and x.payload.definition.environment == [],
            np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten(),
        )
    )


def test_node_building_context_nested_merge():
    """Inner context values override outer ones; all keys are present."""
    action = mock_action((1, 1))
    with NodeMetadataContext(requirements=Requirements(needs_gpu=False, environment=["outer"])):
        with NodeMetadataContext(requirements=Requirements(environment=["middle"]), builder=BuilderMetadata(blockId="test_block")):
            with NodeMetadataContext(requirements=Requirements(needs_gpu=True)):
                mapped_action = action.map(lambda x: x)

    nodes = np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten()
    assert all(n.payload.definition.needs_gpu and n.metadata.requirements.needs_gpu for n in nodes)
    assert all(set(n.payload.definition.environment) == {"middle", "outer"} for n in nodes)
    assert all(n.metadata.builder.blockId == "test_block" for n in nodes)


def test_node_building_context_direct_param_wins():
    """Direct metadata= argument overrides context-provided metadata."""
    action = mock_action((1, 1))
    with NodeMetadataContext(requirements=Requirements(needs_gpu=False, environment=["from_context"])):

        @ekw_mark.add_execution_metadata(needs_gpu=True, environment=["direct"])
        def test_function(x):
            return x

        mapped_action = action.map(test_function)

    nodes = np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten()
    assert all(n.payload.definition.needs_gpu for n in nodes)
    assert all(set(n.payload.definition.environment) == {"direct", "from_context"} for n in nodes)


def test_node_building_context_full_example():
    """Reproduces the docstring example with all three sources combined."""
    action = mock_action((1, 1))
    with NodeMetadataContext(requirements=Requirements(needs_gpu=False), builder=BuilderMetadata(blockId="test_block")):
        with NodeMetadataContext(requirements=Requirements(environment=[]), artifacts=Artifacts(artifact_urls={"test_artifact": "url_1"})):
            with NodeMetadataContext(requirements=Requirements(needs_gpu=True), builder=BuilderMetadata(blockId="inner_block")):
                mapped_action = action.map(
                    lambda x: x,
                    node_metadata=NodeMetadata(
                        requirements=Requirements(environment=["value4"]),
                        artifacts=Artifacts(artifact_urls={"test_artifact": "url_1"}),
                    ),
                )

    nodes = np.atleast_1d(nodetree_array(mapped_action.nodes).values).flatten()
    assert all(n.payload.definition.needs_gpu and n.metadata.requirements.needs_gpu for n in nodes)
    assert all(set(n.payload.definition.environment) == {"value4"} for n in nodes)
    assert all(n.metadata.builder.blockId == "inner_block" for n in nodes)
    assert all(
        n.metadata.artifacts.artifact_urls == {"test_artifact": "url_1"} and n.metadata.artifacts.artifact_urls == n.payload.artifact_urls
        for n in nodes
    )


def test_node_building_context_does_not_bleed_between_sibling_contexts():
    """Sibling contexts do not interfere with each other."""
    action = mock_action((1, 1))
    with NodeMetadataContext(requirements=Requirements(environment=["first"])):
        a1 = action.map(lambda x: x)

    with NodeMetadataContext(requirements=Requirements(environment=["second"])):
        a2 = action.map(lambda x: x)

    assert all(set(n.payload.definition.environment) == {"first"} for n in np.atleast_1d(nodetree_array(a1.nodes).values).flatten())
    assert all(set(n.payload.definition.environment) == {"second"} for n in np.atleast_1d(nodetree_array(a2.nodes).values).flatten())
