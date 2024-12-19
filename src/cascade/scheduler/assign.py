"""
Utility functions for handling assignments -- invocation assumed from scheduler.api module,
for all other purposes this should be treated private
"""

def build_assignment(worker: WorkerId, task: TaskId, state: State) -> Assignment:
    raise NotImplementedError

def assign_within_component(state: State, workers: list[WorkerId], component_idx: int) -> Iterator[Assignment]:
    """Finds a reasonable assignment within a single component. Does not migrate hosts to a different component"""
    component = state.components[component_id]

    # first, attempt optimum-distance assignment
    task_i = 0
    while task_i < len(component.computable):
        task = component.computable[task_i]
        opt_dist = component.computable[task]
        for idx, worker in enumerate(workers):
            if component.worker2task_distance[worker][task] == opt_dist:
                yield build_assignment(worker, task, state)
                worker.pop(idx)
                component.computable.pop(task_i)
                component.core.weight -= 1
                state.idle_workers.remove(worker)
                task_i -= 1
                break
        task_i += 1

    # second, sort task-worker combination by first overhead, second value, and pick greedily
    remaining_t = {t: i for i, t in enumerate(component.computable)}
    remaining_w = set(workers)
    candidates = [
        (state.worker2taskOverhead[w][t], component.core.value[t], w, t)
        for w in workers
        for t in remaining_t
    ]
    candidates.sort(key=lambda e: (e[0], e[1]))
    for _, _, worker, task in candidates:
        if task in remaining_t and worker in remaining_w:
            yield build_assignment(worker, task, state)
            component.computable.pop(remaining_t[task])
            remaining_t.remove(task)
            remaining_w.remove(worker)
            state.idle_workers.remove(worker)
            component.core.weight -= 1

def migrate_to_component(host: HostId, component_id: ComponentId, state: State) -> State:
    """Assuming original component assigned to the host didn't have enough tasks anymore,
    we invoke this function and update state to reflect it"""
    state.host2component[host] = component_idx
    # TODO distances
    raise NotImplementedError

    return State


###

def update_dist4new_computable(worker2task: Worker2TaskDistance, task: TaskId, component: ComponentCore, worker2ds: dict[WorkerId, dict[DatasetId, DatasetStatus]]) -> None:
    raise NotImplementedError
    # TODO we don't currently consider other workers at the host, this is be subopt!
    eligible = {DatasetStatus.preparing, DatasetStatus.available}

    for worker in worker2task.keys():
        worker2task[worker][task] = component.depth
        for ds, state in worker2ds[worker].items():
            if state not in eligible:
                continue
            # TODO we only consider min task distance, whereas weighing by volume/ratio would make more sense
            worker2task[worker][task] = min(worker2task[worker][task], task2task[ds.task][task])



####


