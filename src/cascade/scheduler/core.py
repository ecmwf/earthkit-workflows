# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass

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


@dataclass
class Schedule:
    components: list[ComponentSchedule]
    ts2component: dict[TaskId, ComponentId]
    host2component: dict[HostId, ComponentId | None]
    worker2task_overhead: Worker2TaskDistance

    computable: int  # sum over components.computable

    def has_computable(self) -> bool:
        return self.computable > 0


@dataclass
class Assignment:
    worker: WorkerId
    tasks: list[TaskId]
    prep: list[tuple[DatasetId, HostId]]
    outputs: set[DatasetId]
