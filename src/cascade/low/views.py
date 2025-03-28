"""
Utility functions and transformers for the core graph objects
"""

from collections import defaultdict

from cascade.low.core import DatasetId, JobInstance, Task2TaskEdge, TaskId


def param_source(
    edges: list[Task2TaskEdge],
) -> dict[TaskId, dict[int | str, DatasetId]]:
    """Returns map[sink_task][sink_input] = (source_task, source_output)"""
    rv: dict[TaskId, dict[int | str, DatasetId]] = defaultdict(lambda: defaultdict(lambda: {}))  # type: ignore
    for e in edges:
        sink_input: int | str
        if e.sink_input_kw is not None:
            if e.sink_input_ps is not None:
                raise TypeError
            else:
                sink_input = e.sink_input_kw
        else:
            if e.sink_input_ps is None:
                raise TypeError
            else:
                sink_input = e.sink_input_ps
        rv[e.sink_task][sink_input] = e.source
    return rv


def dependants(edges: list[Task2TaskEdge]) -> dict[DatasetId, set[TaskId]]:
    """Returns map[(source_task, source_output)] = set(sink_task)"""
    rv: dict[DatasetId, set[TaskId]] = defaultdict(set)
    for e in edges:
        rv[e.source].add(e.sink_task)
    return rv


def sinks(job: JobInstance) -> set[DatasetId]:
    non_sinks = {k for k, v in dependants(job.edges).items() if v}
    return {
        dataset
        for task in job.tasks
        for dataset in job.outputs_of(task)
        if dataset not in non_sinks
    }
