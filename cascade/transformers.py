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
