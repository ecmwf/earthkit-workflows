from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, runtime_checkable

from pyrsistent import PSet
from pyrsistent import s as pset
from typing_extensions import Self

from cascade.low.core import Environment, JobExecutionRecord, JobInstance, Schedule
from cascade.low.func import Either


@dataclass
class EnvironmentState:
    """Keeps track of what was computed already and where.
    Used if we are doing a re-schedule (eg after task crashes or early finishes),
    or by iterative scheduling algorithms"""

    datasetAtHost: PSet[tuple[str, tuple[str, str | int]]] = field(default_factory=pset)
    runningTaskAtHost: PSet[tuple[str, str]] = field(default_factory=pset)
    finishedTaskAtHost: PSet[tuple[str, str]] = field(default_factory=pset)
    # TODO add crash record

    def finished_tasks(self) -> set[str]:
        return {e for e, _ in self.finishedTaskAtHost}

    def finishTaskAt(self, host: str, task: str) -> Self:
        return replace(
            self,
            finishedTaskAtHost=self.finishedTaskAtHost.add(
                (
                    host,
                    task,
                )
            ),
        )

    def computeDatasetAt(self, host: str, dataset: tuple[str, str | int]) -> Self:
        return replace(
            self,
            datasetAtHost=self.datasetAtHost.add(
                (
                    host,
                    dataset,
                )
            ),
        )


@runtime_checkable
class Scheduler(Protocol):
    def schedule(
        self,
        job_instance: JobInstance,
        environment: Environment,
        execution_record: JobExecutionRecord | None,
        environment_state: EnvironmentState,
    ) -> Either[Schedule, str]:
        raise NotImplementedError


class ClasslessScheduler:
    def __init__(
        self,
        f: Callable[
            [JobInstance, Environment, JobExecutionRecord | None, EnvironmentState],
            Either[Schedule, str],
        ],
    ) -> None:
        self.f = f

    def schedule(
        self,
        job_instance: JobInstance,
        environment: Environment,
        execution_record: JobExecutionRecord | None,
        environment_state: EnvironmentState,
    ) -> Either[Schedule, str]:
        if not environment.hosts:
            return Either.error("no hosts given")
        return self.f(job_instance, environment, execution_record, environment_state)
