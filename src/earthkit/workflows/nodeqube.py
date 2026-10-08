from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Callable, Sequence, cast, Iterator
import hashlib

from qubed import Qube 


from cascade.low.core import DefaultTaskOutput, TaskDefinition, TaskInstance
from earthkit.workflows.context import _get_context_metadata
from earthkit.workflows.graph import Node as BaseNode
from earthkit.workflows.graph import Output
from earthkit.workflows.metadata import NodeMetadata, Requirements, update_requirements, update_node_metadata
from earthkit.workflows import utils


Coord = tuple[str, list[Any]]
Input = BaseNode | Output
Payload = Callable | str | TaskInstance


def custom_hash(string: str) -> str:
    ret = hashlib.sha256()
    ret.update(string.encode())
    return ret.hexdigest()


def create_task_instance(
    payload: Payload,
    static_input_ps: Optional[list[Any]] = None,
    static_input_kw: Optional[dict[str, Any]] = None,
    requirements: Optional[Requirements] = None,
) -> TaskInstance:
    """
    Create a TaskInstance from a payload.

    Parameters
    ----------
    payload : Payload
        The payload to create the task instance with
    static_input_ps : Optional[list[Any]], optional
        Positional static inputs, by default None. To refer to a node at a particular
        position in the list of inputs, use `Node.Index(index)` where `index` is the
        position of the input node in the list of inputs.
    static_input_kw : Optional[dict[str, Any]], optional
        Keyword static inputs, by default None

    Returns
    -------
    TaskInstance
    """

    requirements = requirements or Requirements()

    if isinstance(payload, TaskInstance):
        update_requirements(requirements, Requirements(environment=payload.definition.environment, needs_gpu=payload.definition.needs_gpu))
        task = payload.model_copy(deep=True)
        task.definition = task.definition.model_copy(update=requirements.model_dump(exclude_none=True))
    elif isinstance(payload, str):
        task = TaskInstance(
            definition=TaskDefinition(
                entrypoint=payload, func=None, input_schema={}, output_schema=[], **requirements.model_dump(exclude_none=True)
            ),
            static_input_ps={str(i): v for i, v in enumerate(static_input_ps or [])},
            static_input_kw=static_input_kw or {},
        )
    else:
        task = TaskInstance(
            definition=TaskDefinition(
                entrypoint="",
                func=TaskDefinition.func_enc(cast(Callable, payload)),
                input_schema={},
                output_schema=[],
                **requirements.model_dump(exclude_none=True),
            ),
            static_input_ps={str(i): v for i, v in enumerate(static_input_ps or [])},
            static_input_kw=static_input_kw or {},
        )
    return task


def _resolve_node_metadata(payload: Payload, node_metadata: Optional[NodeMetadata] = None) -> NodeMetadata:
    metadata: NodeMetadata = NodeMetadata()
    # From mark decorators on functions
    update_requirements(metadata.requirements, Requirements(**getattr(payload, "_cascade", {})))
    update_node_metadata(metadata, _get_context_metadata())
    update_node_metadata(metadata, node_metadata or NodeMetadata())
    if isinstance(payload, TaskInstance):
        update_node_metadata(
            metadata,
            NodeMetadata(
                requirements=Requirements(environment=payload.definition.environment, needs_gpu=payload.definition.needs_gpu),
            ),
        )
    return metadata

class Node(BaseNode):
    @dataclass
    class Index:
        value: int

    def __init__(
        self,
        payload: Payload,
        inputs: Input | Sequence[Input] = [],
        num_outputs: int = 1,
        name: Optional[str] = None,
        metadata: Optional[NodeMetadata] = None,
    ):
        self._for_copy = (payload, inputs, num_outputs)
        if isinstance(inputs, Input):
            inputs = [inputs]
        metadata = _resolve_node_metadata(payload, node_metadata=metadata)
        task = create_task_instance(payload, requirements=metadata.requirements)
        task = task.model_copy(deep=True)
        node_outputs = None if num_outputs == 1 else [f"{x:0{len(str(num_outputs - 1))}d}" for x in range(num_outputs)]
        if len(task.definition.input_schema) == 0:
            task.definition.input_schema = {k: "Any" for k in task.static_input_kw.keys()}
        if len(task.definition.output_schema) == 0:
            task.definition.output_schema = [(e, "Any") for e in node_outputs or [DefaultTaskOutput]]

        # Insert in input nodes not already present in task.static_input_ps
        insert_index = 0
        for i in range(len(inputs)):
            node_index = Node.Index(i)
            if node_index in task.static_input_ps.values():
                continue
            while str(insert_index) in task.static_input_ps:
                insert_index += 1
            task.static_input_ps[str(insert_index)] = node_index

        node_inputs = {}
        for pos, index in task.static_input_ps.items():
            if isinstance(index, Node.Index):
                if index.value >= len(inputs):
                    raise ValueError(f"Node static_input_ps index {index.value} exceeds number of input nodes {len(inputs)}")
                node_inputs[pos] = inputs[index.value]
                task.static_input_ps[pos] = None

        name = name or task.definition.func or task.definition.entrypoint
        name += custom_hash(f"{task}{[x.name if isinstance(x, BaseNode) else f'{x.parent.name}.{x.name}' for x in inputs]}")

        super().__init__(
            name,
            outputs=node_outputs,
            payload=task,
            metadata=metadata,
            **node_inputs,
        )

    def __str__(self) -> str:
        return f"Node {self.name}, inputs: {[x.parent.name for x in self.inputs.values()]}, payload: {self.payload}"

    def copy(self) -> "Node":
        return self.__class__(*self._for_copy)  # type: ignore[arg-type]


class NodeQube:

    def __init__(self, qube: Qube, nodes: dict[str, Node]):
        self.qube = qube
        self.nodes = nodes

    @staticmethod
    def key(datacube: dict[str, Any]) -> str:
        if list(utils.expand(datacube)) != 1:
            raise ValueError("Datacube must contain only single values for each dimension to generate a unique key.")
        return str(sorted(datacube.items()))

    @staticmethod
    def empty() -> NodeQube:
        return NodeQube(Qube.empty(), {})

    @staticmethod
    def datacube(key: str) -> dict[str, Any]:
        return dict(eval(key))

    def is_empty(self) -> bool:
        return self.qube.is_empty()

    def dimensions(self) -> set[str]:
        return self.qube.dimensions()

    def axes(self) -> dict[str, set[str]]:
        return self.qube.axes()

    def datacubes(self, expand: bool = False) -> Iterator[dict[str, Any]]:
        datacubes = self.qube.datacubes()
        if not expand:
            return datacubes
        for datacube in datacubes:
            yield from utils.expand(datacube)

    def get_nodes(self, qube: Qube) -> dict[str, Node]:
        nodes = {}
        for unique_cube in qube.datacubes(expand=True):
            key = self.key(unique_cube)
            nodes[key] = self.nodes[key]
        return nodes

    def _reindex_nodes(self, new_qube: Qube) -> dict[str, Node]:
        new_dimensions = new_qube.dimensions()
        in_new_dims = set(new_dimensions) - set(self.qube.dimensions())
        nodes = {}
        for unique_cube in new_qube.datacubes(expand=True):
            new_key = self.key(unique_cube)
            for key in in_new_dims:
                unique_cube.pop(key, None)
            unique_nodeqube = self.select(unique_cube)
            nodes[new_key] = unique_nodeqube.node()
        return nodes

    def select(self, criteria: dict[str, str]) -> NodeQube:
        selection = self.qube.select(criteria)
        if selection.is_empty():
            raise ValueError(f"No nodes found matching selection criteria: {criteria}")
        return NodeQube(selection, self.get_nodes(selection))

    def iselect(self, criteria: dict[str, str]) -> NodeQube:
        raise NotImplementedError()

    def add_scalar_dimension(self, name: str, value: Any) -> NodeQube:
        new_qube = self.qube.expand({name: [value]})
        return NodeQube(new_qube, self._reindex_nodes(new_qube))

    def drop_scalar_dimension(self, name: str) -> NodeQube:
        values = self.qube.all_unique_dim_coords()[name]
        if len(values) != 1:
            raise ValueError(f"Cannot drop dimension {name} with multiple values: {values}")
        new_qube = self.qube.drop([name])
        return NodeQube(new_qube, self._reindex_nodes(new_qube))

    def append(self, other: NodeQube) -> NodeQube:
        new_qube = self.qube.append(other.qube)
        return NodeQube(new_qube, {**self.nodes, **other.nodes})

    def flatten(self, new_dim: str, keep_dims: Optional[set[str]] = None) -> NodeQube:
        keep_dims = keep_dims or set()
        new_qube = Qube.empty()
        new_nodes = {}
        for datacube in self.datacubes():
            dimensions = set(datacube.keys())
            diff = dimensions - keep_dims
            for index, unique_qube in enumerate(utils.expand(datacube, dims=list(diff))):
                old_key = self.key(unique_qube)
                new_key = self.key(dict({k: v for k, v in unique_qube.items() if k not in diff}, new_dim=index))
                new_nodes[new_key] = self.nodes[old_key]
                datacube.setdefault(new_dim, []).append(index)
            new_qube = new_qube.append(Qube.from_datacube(datacube))
        return NodeQube(new_qube, new_nodes)

    def expand_outputs(self, yields: Optional[Coord] = None) -> NodeQube:
        if yields is None:
            return self
        ydim, ycoords = yields
        new_nodes = {}
        new_qube = self.qube.expand({ydim: ycoords})
        for key, node in self.nodes.items():
            for i, out in enumerate(node.outputs):
                new_key = self.key(dict({ydim: ycoords[i]}, **self.datacube(key)))
                new_nodes[new_key] = node.get_output(out)
        return NodeQube(new_qube, new_nodes)

    def node(self) -> Node:
        if len(self.nodes) != 1:
            raise ValueError(f"NodeQube must contain exactly one node to retrieve it, but contains {len(self.nodes)} nodes.")
        return list(self.nodes.values())[0]
