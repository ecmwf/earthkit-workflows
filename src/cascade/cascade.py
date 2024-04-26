import os
import dill

from .graph import Graph, deduplicate_nodes
from .graph.export import serialise, deserialise
from .contextgraph import ContextGraph
from .schedulers.depthfirst import DepthFirstScheduler
from .schedulers.schedule import Schedule
from .executors.executor import Executor
from .executors.dask import DaskLocalExecutor
from .visualise import visualise
from .profiler import profile


class Cascade:

    def __init__(self, graph: Graph = Graph([])):
        self._graph = graph
        self._schedule = None
        self._executor = None
        self._context_graph = None

    @property
    def executor(self) -> Executor:
        if self._executor is None:
            self._executor = DaskLocalExecutor()
        return self._executor

    @executor.setter
    def executor(self, executor: Executor):
        self._executor = executor
        # reset schedule and context graph as it may not be compatible with new executor
        self._schedule = None
        self._contextgraph = None

    @property
    def context_graph(self) -> ContextGraph:
        if self._context_graph is None:
            self._context_graph = self.executor.create_context_graph()
        return self._context_graph

    @classmethod
    def from_actions(cls, actions):
        graph = Graph([])
        for action in actions:
            graph += action.graph()
        return cls(graph)

    @classmethod
    def from_serialised(cls, filename: str):
        with open(filename, "rb") as f:
            data = dill.load(f)
            return cls(deserialise(data))

    def serialise(self, filename: str):
        data = serialise(self._graph)
        with open(filename, "wb") as f:
            dill.dump(data, f)

    def visualise(self, *args, **kwargs):
        return visualise(self._graph, *args, **kwargs)

    def schedule(self) -> Schedule:
        self._schedule = DepthFirstScheduler().schedule(self._graph, self.context_graph)
        return self._schedule

    def execute(self):
        if self._schedule is not None:
            input = self._schedule
        else:
            input = self._graph
        return self.executor.execute(input, report="report.html")

    def benchmark(self, base_path: os.PathLike):
        results, self._graph = profile(self._graph, base_path, self.executor)
        return results

    def __add__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        return Cascade(deduplicate_nodes(self._graph + other._graph))

    def __iadd__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        self._graph += other._graph
        self._graph = deduplicate_nodes(self._graph)
        self._schedule = None  # doesn't make sense to have a schedule after merging
        return self
