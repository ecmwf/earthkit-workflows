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
from typing import Iterator

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
        last = definition.output_schema[-1][0]
        return last == dataset.output

    def purge_dataset(self, ds: DatasetId) -> Iterator[HostId]:
        """Drop dataset from all tracking structures, yields hosts that should be sent purge command"""
        for host in self.ds2host[ds]:
            self.host2ds[host].pop(ds)
            for worker in self.host2workers[host]:
                if ds in self.worker2ds[worker]:
                    self.worker2ds[worker].pop(ds)
                    self.ds2worker[ds].pop(worker)
            yield host
        self.ds2host.pop(ds)

    # TODO refac pop idle worker, extend ongoing
    def assign_task(self) -> None:
        raise NotImplementedError

    def task_done(self, task: TaskId, worker: WorkerId) -> None:
        self.worker2ts[worker][task] = TaskStatus.succeeded
        self.ts2worker[task][worker] = TaskStatus.succeeded
        if task in self.ongoing[worker]:
            self.ongoing[worker].remove(task)
            self.ongoing_total -= 1
            self.remaining -= 1
        else:
            raise ValueError(f"{task} success but cant remove from `ongoing`")
        if not self.ongoing[worker]:
            self.idle_workers.add(worker)

    def dataset_preparing(self, dataset: DatasetId, worker: WorkerId) -> None:
        # NOTE Currently, these `if`s are necessary because we issue transmit
        # command when host *has* DS but worker does *not*. This ends up no-op,
        # but we totally dont want host state to reset -- it wouldnt recover
        host_s = self.host2ds[worker.host].get(dataset, DatasetStatus.missing)
        if host_s != DatasetStatus.available:
            self.host2ds[worker.host][dataset] = DatasetStatus.preparing
        ds_s = self.ds2host[dataset].get(worker.host, DatasetStatus.missing)
        if ds_s != DatasetStatus.available:
            self.ds2host[dataset][worker.host] = DatasetStatus.preparing
        self.worker2ds[worker][dataset] = DatasetStatus.preparing
        self.ds2worker[dataset][worker] = DatasetStatus.preparing
        # TODO check that there is no invalid transition? Eg, if it already was
        # preparing or available
        # TODO do we want to do anything for the other workers on the same host?
        # Probably not, rather consider host2ds during assignments


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
