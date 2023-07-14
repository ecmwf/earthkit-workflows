from __future__ import annotations

import networkx as nx
import randomname

####################################################################################################


class Task:
    def __init__(self, cost, memory, name=None):
        self.name = name or randomname.get_name()
        self.cost = cost
        self.memory = memory
        self.state = None

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: Task) -> bool:
        return isinstance(other, Task) and hash(self) == hash(other)
    
    def __repr__(self) -> str:
        return self.name

    def __lt__(self, other) -> bool:
        assert isinstance(other, Task)
        return self.name < other.name

class Communication:
    def __init__(self, source, target, size, name=None):
        self.source = source
        self.target = target
        self.name = name or randomname.get_name()
        self.size = size
        self.state = None

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: Communication) -> bool:
        return isinstance(other, Communication) and hash(self) == hash(other)
    
    def __repr__(self) -> str:
        return self.name


class TaskGraph(nx.DiGraph):

    def __init__(self, **attr):
        self.node_dict = {}
        super().__init__(**attr)

    def add_task(self, cost, memory, name=None):
        t = Task(cost, memory, name)
        self.node_dict[t.name] = t
        super().add_node(t)

    def add_comm_edge(self, u_of_edge, v_of_edge, size):
        u_of_edge = self.node_dict[u_of_edge]
        v_of_edge = self.node_dict[v_of_edge]
        super().add_edge(u_of_edge, v_of_edge, obj=Communication(source=u_of_edge, target=v_of_edge, size=size))

    def get_roots(self):
        return [n for n in self if self.in_degree(n) == 0]
    
    def draw(self, filename):
        import matplotlib.pyplot as plt 
        pos = nx.drawing.nx_agraph.graphviz_layout(self, prog='dot', args='-Nshape=box')
        nx.draw_networkx(self, pos=pos, with_labels=False)
        plt.savefig(filename)
        plt.clf()

    def _make_communication_task(self, source, target, edge):
        t = Communication(source, target, edge.size, edge.name)
        self.node_dict[t.name] = t
        super().add_node(t)

        self.remove_edge(source, target)
        super().add_edge(source, t)
        super().add_edge(t, target)
        return t
        

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