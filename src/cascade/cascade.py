
from .graph import Graph
from .contextgraph import ContextGraph
from .schedulers.depthfirst import DepthFirstScheduler
from .schedulers.schedule import Schedule
from .executors.dask import DaskLocalExecutor
from .visualise import visualise


class Cascade:

    def __init__(self, graph, schedule: Schedule = None):
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
    
    def schedule(self, context: ContextGraph = None) -> Schedule:
        self._schedule = DepthFirstScheduler().schedule(self._graph, context)
        return self._schedule

    def execute(self, *args, **kwargs):
        # if self._schedule is None:
        #     self.schedule()
        return DaskLocalExecutor(*args, **kwargs).execute(self._graph)
    
    def __add__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        return Cascade(self.graph + other.graph)

    def __iadd__(self, other: "Cascade") -> "Cascade":
        if not isinstance(other, Cascade):
            return NotImplemented
        self._graph += other._graph
        self._schedule = None  # doesn't make sense to have a schedule after merging
