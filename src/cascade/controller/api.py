"""
Describes the interfaces:
 - Executor: represents a set of workers that can be finely controlled by
   a Controller. Dask implementation is located in this module, Forecast-in-a-Box
   implements its own.
 - Controller: represents an interface for managing job execution. There is one
   implementation in this module, and most likely we won't need another.

Note that not all execution options are based on this Controller interface.
In particular, Dask Delayed is simply accepting JobInstance & Schedule, as its
own controller is under the hood.

The executor instances are not assumed to be re-usable for multiple parallel jobs.
The scheduler will also assumes there is no other workload running on said executor.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
from typing_extensions import Self

from cascade.low.core import Any, Environment, JobInstance, Schedule, TaskInstance
from cascade.low.views import dependants

@dataclass
class VariableWiring:
    """A view of source of an edge, ie, how a TaskInstance obtains a dynamic input"""

    sourceTask: str
    sourceOutput: str
    intoKwarg: Optional[str]
    intoPosition: Optional[int]
    annotation: str


@dataclass
class ExecutableTaskInstance:
    """A wrapper around TaskInstance that contains necessary means of execution"""

    task: TaskInstance
    name: str
    wirings: list[VariableWiring]
    published_outputs: set[str]

@dataclass
class ExecutableSubgraph:
    tasks: list[ExecutableTaskInstance]


@runtime_checkable
class Executor(Protocol):
    # TODO update the is_done, wait_some and return values in favour of just using
    # EnvironmentState as input/output argument
    def get_environment(self) -> Environment:
        """Used by the scheduler to build a schedule"""
        raise NotImplementedError

    def run_at(self, subgraph: ExecutableSubgraph, host: str) -> str:
        """Run a subgraph at the host asap.
        Return a unique id which can be later awaited"""
        raise NotImplementedError

    def scatter(self, taskName: str, outputName: str, hosts: set[str]) -> str:
        """A particular task's output should be asap scattered to listed hosts.
        Return a unique id which can be later awaited."""
        raise NotImplementedError

    def purge(
        self, taskName: str, outputName: str, hosts: set[str] | None = None
    ) -> None:
        """Listed hosts should asap free said dataset. If no hosts given, all-out
        purge commences"""
        raise NotImplementedError

    def fetch_as_url(self, taskName: str, outputName: str) -> str:
        """URL for downloading a particular result. Does not block, result does
        not need to be ready."""
        raise NotImplementedError

    def fetch_as_value(self, taskName: str, outputName: str) -> Any:
        """A particular result as a value. If not ready, blocks. Fetching itself
        is also blocking."""
        raise NotImplementedError

    def wait_some(self, ids: set[str], timeout_sec: int | None = None) -> set[str]:
        """Blocks until at least one of the tasks behind the given ids finishes.
        Returns all ids whose tasks have finished.
        If no timeout given, may block forever."""
        raise NotImplementedError

    def is_done(self, id_: str) -> bool:
        """Non-waiting call to check status of a given activity"""
        raise NotImplementedError


@dataclass
class PurgingPolicy:
    """If eager=true, then delete all outputs once their dependants finish, unless
    they have no descendants at all or they are in `preserve`. If false, no outputs
    are deleted, and `preserve` is ignored."""

    eager: bool = field(default=True)
    preserve: set[tuple[str, str]] = field(default_factory=set)

    @classmethod
    def default(cls, job: JobInstance) -> Self:
        """Sets preserve=True for all sinks of the graph"""
        task_dependants = dependants(job.edges)
        preserve = {
            (task, output)
            for task, instance in job.tasks.items()
            for output in instance.definition.output_schema.keys()
            if not task_dependants[(task, output)]
        }
        return cls(eager=True, preserve=preserve)

@runtime_checkable
class Controller(Protocol):
    def submit(
        self,
        job: JobInstance,
        schedule: Schedule,
        executor: Executor,
        purging_policy: PurgingPolicy,
    ) -> None:
        """Blocks until the whole JobInstance is computed.
        Results should be fetched directly from the executor, then explicitly released
        """
        raise NotImplementedError
