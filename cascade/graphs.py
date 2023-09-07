import randomname
import networkx as nx

from ppgraph import Graph, Transformer, Sink
from ppgraph import Node

####################################################################################################


class Task(Node):
    def __init__(self, name, payload):
        super().__init__(name, payload=payload)
        self.cost = 0
        self.in_memory = 0
        self.out_memory = 0
        self.state = None

    @property
    def memory(self):
        return max(self.in_memory, self.out_memory)


class Communication(Node):
    def __init__(self, name, source, size):
        super().__init__(name, payload=None, input=source)
        self.size = size
        self.state = None


class TaskGraph(Graph):
    def predecessors(self, task) -> list[Task]:
        return [
            x if isinstance(x, Task) else x[0]
            for x in self.get_predecessors(task).values()
        ]

    def successors(self, task) -> list[Task]:
        return [x[0] for x in sum(self.get_successors(task).values(), [])]


class ExecutionGraph(TaskGraph):
    def _make_communication_task(self, source, target):
        t = Communication(f"{source.name}-{target.name}", source, source.out_memory)

        for iname, input in target.inputs.items():
            if input.name == source:
                target.inputs[iname] = t.get_output()
                break
        return t

    def edges(self):
        for node in self.nodes():
            for input in node.inputs.values():
                yield input.parent, node


class _ToTaskGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.payload)
        newnode.inputs = inputs
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> TaskGraph:
        return TaskGraph(sinks)


def to_task_graph(graph: Graph) -> TaskGraph:
    return _ToTaskGraph().transform(graph)


class _ToExecutionGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.payload)
        newnode.inputs = inputs
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> ExecutionGraph:
        return ExecutionGraph(sinks)


def to_execution_graph(graph: Graph) -> ExecutionGraph:
    return _ToExecutionGraph().transform(graph)


####################################################################################################


class Processor:
    def __init__(self, name, type, speed, memory):
        self.name = name
        self.type = type
        self.speed = speed
        self.memory = memory
        # host, port, etc.

    def __hash__(self) -> int:
        return hash(self.name)


class Communicator:
    def __init__(self, source, target, bandwidth, latency):
        self.source = source
        self.target = target
        self.bandwidth = bandwidth
        self.latency = latency
        self.name = randomname.get_name()

    def __hash__(self) -> int:
        return hash(self.source + self.target + str(self.bandwidth) + str(self.latency))


class ContextGraph(nx.Graph):
    def __init__(self, **attr):
        self.node_dict = {}
        super().__init__(**attr)

    def add_node(self, name, type, speed, memory):
        ex = Processor(name, type, speed, memory)
        self.node_dict[ex.name] = ex
        return super().add_node(ex)

    def add_edge(self, u_of_edge, v_of_edge, bandwidth, latency):
        u_of_edge = self.node_dict[u_of_edge]
        v_of_edge = self.node_dict[v_of_edge]
        c = Communicator(u_of_edge, v_of_edge, bandwidth, latency)
        return super().add_edge(u_of_edge, v_of_edge, obj=c)
