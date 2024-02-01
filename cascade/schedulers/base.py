from abc import ABC, abstractmethod

from cascade.graph import Graph
from .schedule import Schedule


class Scheduler(ABC):
    @abstractmethod
    def schedule(self, task_graph: Graph, *args, **kwargs) -> Schedule:
        pass
