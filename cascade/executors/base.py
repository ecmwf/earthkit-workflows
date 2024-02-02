from abc import ABC, abstractmethod
from typing import Any

from cascade.schedulers.schedule import Schedule
from cascade.graph import Graph


class Executor(ABC):
    @abstractmethod
    def execute(self, schedule: Schedule | Graph, **kwargs) -> Any:
        pass
