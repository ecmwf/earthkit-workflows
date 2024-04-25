import os

from .graph import Graph
from .schedulers.depthfirst import DepthFirstScheduler
from .schedulers.schedule import Schedule
from .executors.dask import DaskLocalExecutor
from .visualise import visualise
from .profiler import profile


class Cascade:

    def __init__(self, graph: Graph, schedule: Schedule = None):
        self._graph = graph
        self._schedule = schedule

    @classmethod
    def from_actions(cls, actions):
        graph = Graph([])
        for action in actions:
            graph += action.graph()
        return cls(graph)

    @classmethod
    def from_serialised(cls, serialised):
        return NotImplementedError

    def serialise(self):
        return NotImplementedError

    def visualise(self, *args, **kwargs):
        return visualise(self._graph, *args, **kwargs)

    def schedule(self, *args, **kwargs) -> Schedule:
        executor = DaskLocalExecutor(*args, **kwargs)
        self._schedule = DepthFirstScheduler().schedule(
            self._graph, executor.create_context_graph()
        )
        return self._schedule

    def execute(self, schedule=None, *args, **kwargs):
        if schedule is not None:
            input = schedule
        elif self._schedule is not None:
            input = self._schedule
        else:
            input = self._graph
        return DaskLocalExecutor(*args, **kwargs).execute(input, report="report.html")

    def benchmark(self, base_path: os.PathLike, *args, **kwargs):
        executor = DaskLocalExecutor(*args, **kwargs)
        results, self._graph = profile(self._graph, base_path, executor)
        return results

    def __add__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        return Cascade(self.graph + other.graph)

    def __iadd__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        self._graph += other._graph
        self._schedule = None  # doesn't make sense to have a schedule after merging
