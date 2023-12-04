import randomname
import networkx as nx


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
