# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Callable

from .graph import Graph, Node, Output, Transformer
from .taskgraph import ExecutionGraph, Resources, Task, TaskGraph


class _ToTaskGraph(Transformer):
    def __init__(self, resource_map: dict[str, Resources]):
        self.resource_map = resource_map

    def node(self, node: Node, **inputs: Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        newnode.inputs = inputs
        newnode.resources = self.resource_map.get(node.name, Resources())
        return newnode

    def graph(self, graph: Graph, sinks: list[Node]) -> TaskGraph:
        return TaskGraph(sinks)


def to_task_graph(graph: Graph, resource_map: dict[str, Resources] = {}) -> TaskGraph:
    """Transform graph into task graph, with resource allocation for each task.

    Params
    ------
    graph: Graph to transform
    resource_map: dict of resources for each task

    Returns
    -------
    TaskGraph
    """
    return _ToTaskGraph(resource_map).transform(graph)


class _ToExecutionGraph(Transformer):
    def __init__(self, state: Callable | None = None):
        self.state = state

    def node(self, node: Node, **inputs: Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        if isinstance(node, Task):
            newnode.resources = node.resources
        newnode.inputs = inputs
        newnode.state = self.state() if self.state is not None else None
        return newnode

    def graph(self, graph: Graph, sinks: list[Node]) -> ExecutionGraph:
        return ExecutionGraph(sinks)


def to_execution_graph(graph: Graph, state: Callable | None = None) -> ExecutionGraph:
    return _ToExecutionGraph(state).transform(graph)
