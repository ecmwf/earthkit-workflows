"""
Core datastructures of the scheduler module
"""

# TODO consider putting controller.State here to allow for dynamic recomputations of schedules

from pydantic import BaseModel
from typing import Protocol, runtime_checkable, Iterable
from cascade.low.core import TaskId

class Schedule(BaseModel):
    layers: list[list[TaskId]]

    def schedule_from_layer(self, tasks: Iterable[TaskId], layer: int) -> int|None:
        """Drains the `tasks` and mutates `self` to mark progress. Returns the id
        of the next non-empty layer"""
        for task in tasks:
            self.layers[layer].remove(task)
        while True:
            if layer >= len(self.layers):
                return None
            elif not self.layers[layer]:
                self.layers.pop(layer)
            else:
                return layer
