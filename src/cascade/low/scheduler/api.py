import logging
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Callable, Protocol, runtime_checkable, Iterable, cast

from pyrsistent import PSet
from pyrsistent import s as pset
from typing_extensions import Self

from cascade.low.core import Environment, JobExecutionRecord, JobInstance, Schedule
from cascade.low.func import Either

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """Keeps track of what was computed already and where.
    Used if we are doing a re-schedule (eg after task crashes or early finishes),
    or by iterative scheduling algorithms"""

    datasetAtHost: PSet[tuple[str, tuple[str, str]]] = field(default_factory=pset)
    inProgDatasetAtHost: PSet[tuple[str, tuple[str, str]]] = field(default_factory=pset)
    runningTaskAtHost: PSet[tuple[str, str]] = field(default_factory=pset)
    finishedTaskAtHost: PSet[tuple[str, str]] = field(default_factory=pset)
    # TODO add crash record
    # TODO optimize the structures -- perhaps dict[host][ts|ds] is better

    def finished_tasks(self) -> set[str]:
        return {e for _, e in self.finishedTaskAtHost}

    def started_tasks(self) -> set[str]:
        return {e for _, e in chain(self.finishedTaskAtHost, self.runningTaskAtHost)}

    def available_datasets(self) -> set[tuple[str, str]]:
        return {e for _, e in self.datasetAtHost}

    def hosts_of_ds(self, dataset: tuple[str, str]) -> set[str]:
        return {h for h, d in self.datasetAtHost if d == dataset}

    def ds_of_host(self, host: str, include_in_prog: bool) -> set[tuple[str, str]]:
        computed = {e for h, e in self.datasetAtHost if h == host}
        if include_in_prog:
            in_prog = {e for h, e in self.inProgDatasetAtHost if h == host}
            return computed.union(in_prog)
        else:
            return computed

    def ds_and_ts_of_host(self, host: str) -> tuple[set[tuple[str, str]], set[str]]:
        """Datasets and running tasks"""
        return (
            {e for h, e in self.datasetAtHost if h == host},
            {e for h, e in self.runningTaskAtHost if h == host},
        )

    def runTaskAt(self, host: str, task: str, outputs: Iterable[tuple[str, str]]) -> Self:
        logger.debug(f"running {task=} at {host=}")
        return replace(
            self,
            runningTaskAtHost=self.runningTaskAtHost.add((host, task)),
            inProgDatasetAtHost=self.inProgDatasetAtHost.update((host, e) for e in outputs),
        )

    def finishTaskAt(self, host: str, task: str) -> Self:
        logger.debug(f"finishing {task=} at {host=}")
        return replace(
            self,
            finishedTaskAtHost=self.finishedTaskAtHost.add((host, task)),
            runningTaskAtHost=self.runningTaskAtHost.remove((host, task)),
        )

    def computeDatasetsAt(self, host: str, datasets: Iterable[tuple[str, str]]) -> Self:
        logger.debug(f"computed {datasets=} at {host=}")
        return replace(
            self,
            datasetAtHost=self.datasetAtHost.update((host, dataset) for dataset in datasets),
            inProgDatasetAtHost=self.inProgDatasetAtHost - PSet(datasets), # type: ignore
        )

    def purgeDatasetAt(self, host: str, dataset: tuple[str, str]) -> Self:
        logger.debug(f"purged {dataset=} at {host=}")
        return replace(
            self,
            datasetAtHost=self.datasetAtHost.remove((host, dataset)),
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

@runtime_checkable
class ScheduleTransformer(Protocol):
    def transform(self, schedule: Schedule, record: JobExecutionRecord|None) -> Schedule:
        raise NotImplementedError
