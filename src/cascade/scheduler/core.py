# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from enum import Enum
from typing import Any

from cascade.low.core import DatasetId, HostId, TaskId, WorkerId

Task2TaskDistance = dict[TaskId, dict[TaskId, int]]

TaskValue = dict[TaskId, int]


@dataclass
class ComponentCore:
    nodes: list[TaskId]
    sources: list[TaskId]
    distance_matrix: Task2TaskDistance  # nearest common descendant
    value: TaskValue  # closer to a sink -> higher value
    depth: int  # maximum value

    def weight(self) -> int:
        # TODO eventually replace with runtime sum or smth
        return len(self.nodes)


@dataclass
class Preschedule:
    components: list[ComponentCore]  # sorted desc by weight
    edge_o: dict[DatasetId, set[TaskId]]
    edge_i: dict[TaskId, set[DatasetId]]
    task_o: dict[TaskId, set[DatasetId]]


Worker2TaskDistance = dict[WorkerId, dict[TaskId, int]]

ComponentId = int


@dataclass
class ComponentSchedule:
    core: ComponentCore
    weight: int  # of *remaining* tasks -- decreases over time
    computable: dict[TaskId, int]  # task & optimum distance attained by some worker
    # set at build time to contain all inputs for every task, gradually removed in controller.notify as inputs are
    # being computed, to facilitate fast filling of the `computable`. Can be seen as aggregation & map of ds2worker
    is_computable_tracker: dict[TaskId, set[DatasetId]]
    # w2t_dist generally holds values for all workers of hosts assigned to this component and for all
    # tasks that are either computable or that are among outputs of currently prepared tasks (as those
    # could become computable without any further planning)
    worker2task_distance: Worker2TaskDistance
    # eligible values -- a cached value. Used when migrating new workers to the component, inserted whenever a parent of this task gets `preparing`, removed when this task is made computable
    worker2task_values: set[TaskId]


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
class State:
    """Captures what is where -- datasets, running tasks, ... Used for decision making and progress tracking"""

    # NOTE there is some leak of controller's trackers in here... but the whole scheduler-controller-executorFacade
    # separation is getting quite weird

    # lookups
    edge_o: dict[DatasetId, set[TaskId]]
    edge_i: dict[TaskId, set[DatasetId]]
    task_o: dict[TaskId, set[DatasetId]]
    worker2ds: dict[WorkerId, dict[DatasetId, DatasetStatus]]
    ds2worker: dict[DatasetId, dict[WorkerId, DatasetStatus]]
    ts2worker: dict[TaskId, dict[WorkerId, TaskStatus]]
    worker2ts: dict[WorkerId, dict[TaskId, TaskStatus]]
    host2ds: dict[HostId, dict[DatasetId, DatasetStatus]]
    ds2host: dict[DatasetId, dict[HostId, DatasetStatus]]

    # schedule -- updated by scheduler.api.{assign, plan}, except `computable` and `components.computable` which controller.notify updates
    components: list[ComponentSchedule]
    ts2component: dict[TaskId, ComponentId]
    host2component: dict[HostId, ComponentId | None]
    host2workers: dict[HostId, list[WorkerId]]
    computable: int
    remaining: int
    total: int
    worker2task_overhead: Worker2TaskDistance

    # trackers
    # add by controller.notify, remove by scheduler.api.assign
    idle_workers: set[WorkerId]
    # add by scheduler.api.plan, remove by controller.notify. A projection of worker2ts for running only
    ongoing: dict[WorkerId, set[TaskId]]
    ongoing_total: int  # view of the above
    # NOTE the purging_tracker is also used in `consider_computable` and `api.plan` -- come up with a better name! Or separate lookup from tracker?
    # add by scheduler.api.initialize, remove by controller.notify
    purging_tracker: dict[DatasetId, set[TaskId]]
    # add by controller.act post-fetch and by controller.notify, removed by controller.act.
    # TODO extend with `at`, for fine graining?
    purging_queue: list[DatasetId]
    # key add by scheduler.api.init, value add by controller.notify
    outputs: dict[DatasetId, Any]
    # add by controller.notify, remove by controller.act
    fetching_queue: dict[DatasetId, HostId]


def has_computable(state: State) -> bool:
    return state.computable > 0


def has_awaitable(state: State) -> bool:
    # TODO replace the None in outputs with check on fetch queue (but change that from binary to ternary first)
    return state.ongoing_total > 0 or (None in state.outputs.values())


@dataclass
class Assignment:
    worker: WorkerId
    tasks: list[TaskId]
    prep: list[tuple[DatasetId, HostId]]
    outputs: set[DatasetId]
