"""
A line of numpy operations
"""

from base import JobSpec  # ty:ignore[unresolved-import]

from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import JobInstance, JobInstanceRich


def job() -> JobInstanceRich:
    b = JobBuilder()
    b = b.with_node("source", TaskBuilder.from_entrypoint("runtime.source_numpy", {}, "numpy.ndarray", ["numpy"]))
    b = b.with_node(
        "t1", TaskBuilder.from_entrypoint("runtime.transform_numpy", {"a": "numpy.ndarray"}, "numpy.ndarray", ["numpy"])
    ).with_edge("source", "t1", "a")
    return JobInstanceRich(jobInstance=b.build().get_or_raise(), checkpointSpec=None)


def spc() -> JobSpec:
    return JobSpec(workers=1, hosts=1)
