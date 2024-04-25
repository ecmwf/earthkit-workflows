import networkx as nx
from typing import Iterator


class Processor:
    def __init__(
        self, name: str, type: str, speed: float, memory: float, uri: str = None
    ):
        self.name = name
        self.type = type
        self.speed = speed
        self.memory = memory
        self.uri = uri
        # host, port, etc.

    def __hash__(self) -> int:
        return hash(self.name)


class Communicator:
    def __init__(
        self, source: Processor, target: Processor, bandwidth: float, latency: float
    ):
        self.source = source.name
        self.target = target.name
        self.bandwidth = bandwidth
        self.latency = latency
        self.name = f"{self.source}-{self.target}"

    def __hash__(self) -> int:
        return hash(self.source + self.target + str(self.bandwidth) + str(self.latency))


class ContextGraph(nx.Graph):
    def __init__(self, **attr):
        self.node_dict = {}
        super().__init__(**attr)

    def add_node(
        self, name: str, type: str, speed: float, memory: float, uri: str = None
    ):
        ex = Processor(name, type, speed, memory, uri)
        self.node_dict[ex.name] = ex
        super().add_node(ex)

    def add_edge(
        self, u_of_edge: str, v_of_edge: str, bandwidth: float, latency: float
    ):
        u_of_edge = self.node_dict[u_of_edge]
        v_of_edge = self.node_dict[v_of_edge]
        c = Communicator(u_of_edge, v_of_edge, bandwidth, latency)
        super().add_edge(u_of_edge, v_of_edge, obj=c)

    def communicator(self, u_of_edge: str, v_of_edge: str) -> Communicator:
        """
        Get communicator for edge

        Params
        ------
        u_of_edge: str, source processor name
        v_of_edge: str, target processor name

        Returns
        -------
        Communicator for edge
        """
        return super().get_edge_data(u_of_edge, v_of_edge)["obj"]

    def communicators(self) -> Iterator[Communicator]:
        """
        Iterator over communicators in edges of graphs

        Returns
        -------
        Iterator[Communicator]
        """
        for _, _, communicator in self.edges(data=True):
            yield communicator["obj"]

    def visualise(self, dest: str = "contextgraph.html"):
        from cascade.visualise import visualise_contextgraph

        visualise_contextgraph(self, dest)
