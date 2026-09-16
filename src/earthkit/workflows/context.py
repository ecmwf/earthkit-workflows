import threading
import types
from typing import Any

from earthkit.workflows.metadata import Artifacts, BuilderMetadata, NodeMetadata, Requirements, update_node_metadata

_node_context = threading.local()


def _get_context_stack() -> list[NodeMetadata]:
    if not hasattr(_node_context, "earthkit_workflow_node_metadata_stack"):
        _node_context.earthkit_workflow_node_metadata_stack = []
    return _node_context.earthkit_workflow_node_metadata_stack  # type: ignore[return-value]


def _invalidate_context_cache() -> None:
    if hasattr(_node_context, "earthkit_workflow_node_metadata_resolved"):
        del _node_context.earthkit_workflow_node_metadata_resolved


def _get_context_metadata() -> NodeMetadata:
    if hasattr(_node_context, "earthkit_workflow_node_metadata_resolved"):
        return _node_context.earthkit_workflow_node_metadata_resolved
    result: NodeMetadata = NodeMetadata()
    for frame in _get_context_stack():
        update_node_metadata(result, frame)
    _node_context.earthkit_workflow_node_metadata_resolved = result
    return result


def _pop_context_stack() -> None:
    _invalidate_context_cache()
    _get_context_stack().pop()


class NodeMetadataContext:
    """Context manager that injects metadata into every Node created within it.

    Contexts can be nested; inner values override outer ones on key collision.
    Metadata passed directly to Node overrides any context-provided metadata.
    But for 'environment' types, instead of override we append, as that makes more sense.

    Example
    -------
    with NodeMetadataContext(requirements={"environment": ["my_env"]}, artifacts={}, builder={}):
        action1 = from_source(...)
        action2 = action1.map(some_func)
    """

    def __init__(
        self, requirements: Requirements | None = None, artifacts: Artifacts | None = None, builder: BuilderMetadata | None = None
    ) -> None:
        self._metadata: NodeMetadata = NodeMetadata(
            requirements=requirements or Requirements(),
            artifacts=artifacts or Artifacts(),
            builder=builder or BuilderMetadata(),
        )

    def __enter__(self) -> "NodeMetadataContext":
        _get_context_stack().append(self._metadata)
        _invalidate_context_cache()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        _pop_context_stack()