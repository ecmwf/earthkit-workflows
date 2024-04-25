from typing import Any
from cascade.contextgraph import ContextGraph
from cascade.graph import Graph
from cascade.schedulers.schedule import Schedule
import abc


class Executor(abc.ABC):

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def execute(self, schedule: Graph | Schedule, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def benchmark(self, schedule: Graph | Schedule, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def create_context_graph(self) -> ContextGraph:
        raise NotImplementedError
