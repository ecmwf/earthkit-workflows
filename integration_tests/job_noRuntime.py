"""Needs no runtime at all -- the simplest possible job to run"""

from base import JobSpec  # ty:ignore[unresolved-import]

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance, JobInstanceRich


def job() -> JobInstanceRich:
    b = JobBuilder()
    b = b.with_node("source", TaskBuilder.from_entrypoint("cascade.executor.platform.get_bindabble_self", {}, "str", []))
    return JobInstanceRich(jobInstance=b.build().get_or_raise(), checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)
