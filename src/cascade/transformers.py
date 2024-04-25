from dask.utils import apply

from .graph import Sink, Node, Graph, Transformer
from .taskgraph import Resources, Task, TaskGraph, ExecutionGraph
from .executors.processpool import ProcessPoolExecutor


class _ToTaskGraph(Transformer):
    def __init__(self, resource_map: dict[str, Resources]):
        self.resource_map = resource_map

    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        newnode.inputs = inputs
        newnode.resources = self.resource_map.get(node.name, Resources())
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> TaskGraph:
        return TaskGraph(sinks)


def to_task_graph(graph: Graph, resource_map: dict[str, Resources] = {}) -> TaskGraph:
    """
    Transform graph into task graph, with resource allocation for each task.

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

    def __init__(self, state: callable = None):
        self.state = state

    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        if isinstance(node, Task):
            newnode.resources = node.resources
        newnode.inputs = inputs
        newnode.state = self.state() if self.state is not None else None
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> ExecutionGraph:
        return ExecutionGraph(sinks)


def to_execution_graph(graph: Graph, state: callable = None) -> ExecutionGraph:
    return _ToExecutionGraph(state).transform(graph)


class _ToDaskGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Node:
        new_payload = list(node.payload)
        new_payload[1] = list(new_payload[1])
        for input_name, input in inputs.items():
            if input_name in new_payload[1]:
                new_payload[1][new_payload[1].index(input_name)] = input.parent.name
        newnode = node.copy()
        newnode.payload = tuple([apply] + new_payload)  # Need apply for kwargs
        newnode.inputs = inputs

        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> dict:
        new_graph = Graph(sinks)
        ret = {}
        new_nodes = list(new_graph.nodes(forwards=True))
        for node in new_nodes:
            assert node.name not in ret, f"{node.name} already exists"
            ret[node.name] = node.payload
        assert list(ret.keys()) == [
            node.name for node in new_nodes
        ], f"Expected {len(new_nodes)} nodes, got {len(ret.keys())}"
        return ret


def to_dask_graph(graph: Graph) -> dict:
    return _ToDaskGraph().transform(graph)
