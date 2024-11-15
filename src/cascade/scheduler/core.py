"""
Core datastructures of the scheduler module
"""

# TODO consider putting controller.State here to allow for dynamic recomputations of schedules

from pydantic import BaseModel, model_validator, Field
from typing import Protocol, runtime_checkable, Iterable, Callable, Any
from cascade.low.core import TaskId, JobInstance, JobExecutionRecord, JobExecutionRecord
from cascade.low.func import Either

class Schedule(BaseModel):
    """Represents graph decomposition for scheduling. Independent of number of workers -- the assignment
    of tasks to workers is handled by `controller.plan`.
    When handed over to `controller`, becomes a mutable progress tracker -- thus if the original schedule
    needs to be preserved, make a deep copy."""

    # NOTE this doesnt make that much sense anymore since we combine bfs and dfs. Ideally just reuse the structure
    # for purging datasets to allow for updating here
    layers: list[list[TaskId]]
    computable: list[TaskId] = Field(default_factory=list) # initialised default to 0th layer, then updated by planner
    record: JobExecutionRecord

    @model_validator(mode='before')
    @classmethod
    def init_computable(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get('computable', None):
            data['computable'] = data['layers'].pop(0)
        return data



Scheduler = Callable[[JobInstance, JobExecutionRecord, set[TaskId]], Either[Schedule, str]]
