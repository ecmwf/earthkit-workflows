# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Common data structures and utility methods that form the interface between scheduler and controller.
Primarily manifesting in the JobExecutionContext class -- a proto-scheduler of sorts
"""


from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from cascade.low.core import (
    DatasetId,
    Environment,
    HostId,
    JobInstance,
    TaskId,
    WorkerId,
)


class DatasetStatus(int, Enum):
    missing = -1  # virtual default status, never stored
    preparing = 0  # set by controller
    available = 1  # set by executor
    purged = 2  # temporal command status used as local comms between controller.act and controller.state


class TaskStatus(int, Enum):
    enqueued = 0  # set by controller
    running = 1  # set by executor
    succeeded = 2  # set by executor
    failed = 3  # set by executor


@dataclass
class JobExecutionContext:
    """Captures what is where -- datasets, running tasks, ... Used for decision making and progress tracking.
    Broad interface between (generic) scheduler and controller
    """

    # static
    job_instance: JobInstance
    edge_o: dict[DatasetId, set[TaskId]]
    edge_i: dict[TaskId, set[DatasetId]]
    task_o: dict[TaskId, set[DatasetId]]
    environment: Environment

    # dynamic
    worker2ds: dict[WorkerId, dict[DatasetId, DatasetStatus]]
    ds2worker: dict[DatasetId, dict[WorkerId, DatasetStatus]]
    ts2worker: dict[TaskId, dict[WorkerId, TaskStatus]]
    worker2ts: dict[WorkerId, dict[TaskId, TaskStatus]]
    host2ds: dict[HostId, dict[DatasetId, DatasetStatus]]
    ds2host: dict[DatasetId, dict[HostId, DatasetStatus]]
    host2workers: dict[HostId, list[WorkerId]]

    # aggregations
    idle_workers: set[WorkerId]  # all workers such that worker2ts is empty
    ongoing: dict[WorkerId, set[TaskId]]  # like worker2ts where value is `running`
    ongoing_total: int  # sum of ongoing
    total: int  # size of JobInstance
    remaining: int  # total - sum(tasks that are in `succeeded`)

    def has_awaitable(self) -> bool:
        return self.ongoing_total > 0 or self.remaining > 0

    def is_last_output_of(self, dataset: DatasetId) -> bool:
        """For single-output tasks, always true. For generator tasks, true for the last one.
        Generic KV outputs not supported -- this method wouldnt make any sense.
        """
        definition = self.job_instance.tasks[dataset.task].definition
        # TODO dont sort on each invoke -- precompute
        last = sorted(definition.output_schema.keys())[-1]
        return last == dataset.output

    # TODO refac pop idle worker, extend ongoing
    def assign_task(self) -> None:
        raise NotImplementedError

    # TODO refac mutate ongoing, ongoing total
    def finish_task(self) -> None:
        raise NotImplementedError


def init_context(
    environment: Environment,
    job_instance: JobInstance,
    edge_o: dict[DatasetId, set[TaskId]],
    edge_i: dict[TaskId, set[DatasetId]],
) -> JobExecutionContext:
    host2workers: dict[HostId, list[WorkerId]] = defaultdict(list)
    for worker in environment.workers:
        host2workers[worker.host].append(worker)
    task_o = {task: job_instance.outputs_of(task) for task in job_instance.tasks.keys()}
    total = len(job_instance.tasks.keys())
    return JobExecutionContext(
        job_instance=job_instance,
        edge_o=edge_o,
        edge_i=edge_i,
        task_o=task_o,
        environment=environment,
        worker2ds=defaultdict(dict),
        ds2worker=defaultdict(dict),
        ts2worker=defaultdict(dict),
        worker2ts=defaultdict(dict),
        host2ds=defaultdict(dict),
        ds2host=defaultdict(dict),
        host2workers=host2workers,
        idle_workers=set(environment.workers.keys()),
        ongoing=defaultdict(set),
        ongoing_total=0,
        total=total,
        remaining=total,
    )
