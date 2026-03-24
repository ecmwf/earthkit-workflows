"""
A graph where each node requires a different version of numpy.
An environment with a single host.

This validates that venv corruption is not happening -- ie, a previous
task's pip installs can be erased reliably when a new task with
conflicting requirements is executed.
"""

from base import JobSpec  # ty:ignore[unresolved-import]

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance, JobInstanceRich


def job() -> JobInstanceRich:
    fac = lambda version: TaskBuilder.from_entrypoint(
        "runtime.check_numpy_version",
        {"expected": "str"},
        "bool",
        [f"numpy=={version}"],
    ).with_values(expected=version)

    ji = JobBuilder().with_node("t1", fac("2.0.1")).with_node("t2", fac("2.4.1")).with_node("t3", fac("2.4.2")).build().get_or_raise()
    return JobInstanceRich(jobInstance=ji, checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)
