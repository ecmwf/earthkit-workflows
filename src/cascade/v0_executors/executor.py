import abc
from typing import Any

from cascade.v0_contextgraph import ContextGraph
from cascade.graph import Graph
from cascade.v0_schedulers.schedule import Schedule


class Executor(abc.ABC):
    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def execute(self, schedule: Graph | Schedule, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def create_context_graph(self) -> ContextGraph:
        raise NotImplementedError
