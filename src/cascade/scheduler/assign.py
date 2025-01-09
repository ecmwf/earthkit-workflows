"""
Utility functions for handling assignments -- invocation assumed from scheduler.api module,
for all other purposes this should be treated private
"""

from collections import defaultdict
import logging
from time import perf_counter_ns
from typing import Iterator

from cascade.low.core import WorkerId, DatasetId, TaskId, HostId
from cascade.low.tracing import trace, Microtrace
from cascade.scheduler.core import State, Assignment, DatasetStatus, Worker2TaskDistance, ComponentId, Task2TaskDistance

logger = logging.getLogger(__name__)

def build_assignment(worker: WorkerId, task: TaskId, state: State) -> Assignment:
    eligible_load = {DatasetStatus.preparing, DatasetStatus.available}
    eligible_transmit = {DatasetStatus.available}
    prep: list[tuple[DatasetId, HostId]] = []
    for dataset in state.edge_i[task]:
        at_worker = state.worker2ds[worker]
        if at_worker.get(dataset, DatasetStatus.missing) not in eligible_load:
            if state.host2ds[worker.host].get(dataset, DatasetStatus.missing) in eligible_load:
                # NOTE this currently leads to no-op, but with persistent workers would possibly allow an early fetch
                prep.append((dataset, worker.host))
            else:
                if any(candidate := host for host, status in state.ds2host[dataset].items() if status in eligible_transmit):
                    prep.append((dataset, candidate))
                    # NOTE this is a slight hack, to prevent issuing further transmit commands of this ds to this host
                    # in this phase. A proper state transition happens later in the `plan` phase. We may want to instead
                    # create a new `transmit_queue` state field to capture this, and consume it later during plan
                    state.host2ds[worker.host][dataset] = DatasetStatus.preparing
                    state.ds2host[dataset][worker.host] = DatasetStatus.preparing
                else:
                    raise ValueError(f"{dataset=} not found in any host, whoa whoa!")
            
                
    return Assignment(
        worker=worker,
        tasks=[task], # TODO eager fusing for outdeg=1? Or heuristic via ratio of outdeg vs workers@component?
        prep=prep,
        outputs={ # TODO trim for only the necessary ones
            ds
            for ds in state.task_o[task]
        },
    )

def assign_within_component(state: State, workers: list[WorkerId], component_id: ComponentId) -> Iterator[Assignment]:
    """Finds a reasonable assignment within a single component. Does not migrate hosts to a different component"""
    start = perf_counter_ns()
    component = state.components[component_id]

    # first, attempt optimum-distance assignment
    task_i = 0
    computable_keys = list(component.computable.keys()) # we need to copy because we mutate
    for task in computable_keys:
        opt_dist = component.computable[task]
        for idx, worker in enumerate(workers):
            if component.worker2task_distance[worker][task] == opt_dist:
                end = perf_counter_ns()
                trace(Microtrace.ctrl_assign, end-start)
                yield build_assignment(worker, task, state)
                start = perf_counter_ns()
                workers.pop(idx)
                component.computable.pop(task)
                component.worker2task_values.remove(task)
                component.weight -= 1
                state.computable -= 1
                state.idle_workers.remove(worker)
                break

    # second, sort task-worker combination by first overhead, second value, and pick greedily
    remaining_t = set(component.computable.keys())
    remaining_w = set(workers)
    candidates = [
        (state.worker2task_overhead[w][t], component.core.value[t], w, t)
        for w in workers
        for t in remaining_t
    ]
    candidates.sort(key=lambda e: (e[0], e[1]))
    for _, _, worker, task in candidates:
        if task in remaining_t and worker in remaining_w:
            end = perf_counter_ns()
            trace(Microtrace.ctrl_assign, end-start)
            yield build_assignment(worker, task, state)
            start = perf_counter_ns()
            component.computable.pop(task)
            component.worker2task_values.remove(task)
            remaining_t.remove(task)
            remaining_w.remove(worker)
            state.idle_workers.remove(worker)
            state.computable -= 1
            component.weight -= 1

    end = perf_counter_ns()
    trace(Microtrace.ctrl_assign, end-start)

def update_worker2task_distance(component_id: ComponentId, task: TaskId, worker: WorkerId, state: State) -> State:
    """For a given task and worker, consider all tasks at the worker and see if any attains a better distance to said
    task. If additionally the task is _already_ computable and the global minimum attained by `component.computable`
    is improved, set that too."""
    # TODO we don't currently consider other workers at the host, probably subopt! Ultimately,
    # we need the `assign_within_component` to take both overhead *and* distance into account
    # simultaneously
    worker2task = state.components[component_id].worker2task_distance
    task2task = state.components[component_id].core.distance_matrix
    eligible = {DatasetStatus.preparing, DatasetStatus.available}
    state.components[component_id].worker2task_values.add(task)
    computable = state.components[component_id].computable
    for ds_key, ds_status in state.worker2ds[worker].items():
        if ds_status not in eligible:
            continue
        if state.ts2component[ds_key.task] != component_id:
            continue
        # TODO we only consider min task distance, whereas weighing by volume/ratio would make more sense
        val = min(
            worker2task[worker][task],
            task2task[ds_key.task][task],
        )
        worker2task[worker][task] = val
        if ((current := computable.get(task, None)) is not None) and current > val:
            computable[task] = val

    return state

def set_worker2task_overhead(state: State, worker: WorkerId, task: TaskId) -> State:
    # NOTE beware this is used in migrate host2component as well as twice in notify. We may
    # want to later distinguish between `calc_new` (for migrate and new computable) vs
    # `calc_update` (basicaly when host2host transmit finishes)
    # TODO replace the numerical heuristic here with some numbers based on transfer speeds
    # and dataset volumes
    overhead = 0
    for ds in state.edge_i[task]:
        workerState = state.worker2ds[worker].get(ds, DatasetStatus.missing) 
        if workerState == DatasetStatus.available:
            continue
        if workerState == DatasetStatus.preparing:
            overhead += 1
            continue
        hostState = state.host2ds[worker.host].get(ds, DatasetStatus.missing) 
        if hostState == DatasetStatus.available or hostState == DatasetStatus.preparing:
            overhead += 10
            continue
        overhead += 100
    state.worker2task_overhead[worker][task] = overhead
    return state

def migrate_to_component(host: HostId, component_id: ComponentId, state: State) -> State:
    """Assuming original component assigned to the host didn't have enough tasks anymore,
    we invoke this function and update state to reflect it"""
    state.host2component[host] = component_id
    component = state.components[component_id]
    logger.debug(f"migrate {host=} to {component_id=} => {component.worker2task_values=}")
    for worker in state.host2workers[host]:
        component.worker2task_distance[worker] = defaultdict(lambda : component.core.depth)
        for task in component.worker2task_values:
            state = update_worker2task_distance(component_id, task, worker, state)
            state = set_worker2task_overhead(state, worker, task)

    return state
