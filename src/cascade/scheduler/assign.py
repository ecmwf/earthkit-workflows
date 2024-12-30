"""
Utility functions for handling assignments -- invocation assumed from scheduler.api module,
for all other purposes this should be treated private
"""

from collections import defaultdict
from time import perf_counter_ns
from typing import Iterator

from cascade.low.core import WorkerId, DatasetId, TaskId, HostId
from cascade.low.tracing import trace, Microtrace
from cascade.scheduler.core import State, Assignment, DatasetStatus, Worker2TaskDistance, ComponentId, Task2TaskDistance

def build_assignment(worker: WorkerId, task: TaskId, state: State) -> Assignment:
    eligible_worker = {DatasetStatus.preparing, DatasetStatus.available}
    eligible_host = {DatasetStatus.available}
    prep: list[tuple[DatasetId, HostId]] = []
    for dataset in state.edge_i[task]:
        at_worker = state.worker2ds[worker]
        if at_worker.get(dataset, DatasetStatus.missing) not in eligible_worker:
            if state.host2ds[worker.host].get(dataset, DatasetStatus.missing) in eligible_host:
                prep.append((dataset, worker.host))
            else:
                if any(candidate := host for host, status in state.ds2host[dataset].items() if status in eligible_host):
                    prep.append((dataset, candidate))
                else:
                    raise ValueError(f"{dataset =} not found in any host, whoa whoa!")
                
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
            remaining_t.remove(task)
            remaining_w.remove(worker)
            state.idle_workers.remove(worker)
            state.computable -= 1
            component.weight -= 1

    end = perf_counter_ns()
    trace(Microtrace.ctrl_assign, end-start)

def update_worker2task_distance(worker2task: Worker2TaskDistance, task2task: Task2TaskDistance, task: TaskId, worker: WorkerId, state: State) -> State:
    # TODO we don't currently consider other workers at the host, probably subopt! Ultimately,
    # we need the `assign_within_component` to take both overhead *and* distance into account
    # simultaneously
    eligible = {DatasetStatus.preparing, DatasetStatus.available}
    for ds_key, ds_status in state.worker2ds[worker].items():
        if ds_status not in eligible:
            continue
        # TODO we only consider min task distance, whereas weighing by volume/ratio would make more sense
        worker2task[worker][task] = min(
            worker2task[worker][task],
            task2task[ds_key.task][task],
        )

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
    for task, current_opt in component.computable.items():
        for worker in state.host2workers[host]:
            component.worker2task_distance[worker] = defaultdict(lambda : component.core.depth)
            state = update_worker2task_distance(component.worker2task_distance, component.core.distance_matrix, task, worker, state)
            state = set_worker2task_overhead(state, worker, task)

    return state
