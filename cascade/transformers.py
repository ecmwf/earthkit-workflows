from ppgraph import Sink, Node, Graph, Transformer

from .graphs import Task, TaskGraph, ExecutionGraph


class _ToTaskGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        newnode.inputs = inputs
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> TaskGraph:
        return TaskGraph(sinks)


def to_task_graph(graph: Graph) -> TaskGraph:
    return _ToTaskGraph().transform(graph)


class _ToExecutionGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = node.copy()
        newnode.inputs = inputs
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> ExecutionGraph:
        return ExecutionGraph(sinks)


def to_execution_graph(graph: Graph) -> ExecutionGraph:
    return _ToExecutionGraph().transform(graph)


class _ToDaskGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Node:
        new_payload = list(node.payload)
        for input_name, input in inputs.items():
            if input_name in new_payload:
                new_payload[new_payload.index(input_name)] = input.parent.name
        newnode = node.copy()
        newnode.payload = tuple(new_payload)
        newnode.inputs = inputs

        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> dict:
        new_graph = Graph(sinks)
        ret = {}
        new_nodes = list(new_graph.nodes(forwards=True))
        for node in new_nodes:
            ret[node.name] = node.payload
        assert list(ret.keys()) == [
            node.name for node in new_nodes
        ], f"Expected {len(new_nodes)} for {len(ret.keys())}"
        return ret


def to_dask_graph(graph: Graph) -> dict:
    return _ToDaskGraph().transform(graph)
