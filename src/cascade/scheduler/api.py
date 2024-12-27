from cascade.scheduler.core import State, Preschedule, ComponentSchedule, ComponentId
from cascade.low.core import Environment, WorkerId, DatasetId
from cascade.scheduler.assign import assign_within_component, migrate_to_component
from cascade.low.tracing import timer, Microtrace

def initialize(environment: Environment, preschedule: Preschedule, outputs: set[DatasetId]) -> State:
    """Initializes State based on Preschedule and Environment. Assigns hosts to components"""
    purging_tracker = {
        ds: {task for task in dependants}
        for ds, dependants in preschedule.edge_o.items()
    }

    components: list[ComponentSchedule] = []
    host2component: dict[HostId, ComponentId] = {}
    host2workers: dict[HostId, list[WorkerId]] = defaultdict(list) 
    for worker in environment.workers:
        host2workers[worker.host].append(worker)

    computable = 0
    for componentId, precomponent in enumerate(preschedule.components):
        component = ComponentSchedule(
            core=precomponent,
            computable={task: 0 for task in precomponent.sources},
            worker2task_distance={},
            is_computable_tracker={
                task: {inp for inp in preschedule.edge_i[task]}
                for task in precomponent.nodes
            },
        )
        components.append(component)
        computable += len(precomponent.sources)

    return State(
        edge_o=preschedule.edge_o,
        edge_i=preschedule.edge_i,
        task_o=preschedule.task_o,
        components=components,
        host2component=host2component,
        host2workers=host2workers,
        purging_tracker=purging_tracker,
        outputs={e: None for e in outputs},
        worker_colocations={worker: set(host2workers[worker]) for worker in environment.workers},
        computable=computable,
    )

def assign(state: State) -> Iterator[Assignment]:
    """Given idle workers in `state`, assign actions to workers. Mutates the state:
     - pops from computable & idle workers,
     - decreases weight,
     - changes host2component.
    Yields, to allow for immediate async sending to workers.
    Performance critical section, we need to output an assignment asap. Steps taking longer
    should be deferred to `plan`"""

    # step I: assign within existing components
    components = defaultdict(list)
    for worker in state.idle_workers:
        if (component := state.host2component[worker.host]) is not None:
            components[state.host2component[worker.host]].append(worker)

    for component_id, local_workers in components.items():
        if local_workers:
            yield from assign_within_component(state, local_workers, component_id)
    
    if not state.idle_workers:
        return

    # step II: assign remaining workers to new components
    components = [
        (component.weight, component_id) for component_id, component in state.components
        if component.weight > 0
    ]
    components.sort(reverse = True) # TODO consider number of currently assigned workers too
    migrants = defaultdict(list)
    for worker in remaining_w:
        # TODO we dont currently allow partial assignments, this is subopt!
        if state.components[state.host2component[worker.host]].weight == 0:
            migrants[worker.host].append(worker)
        # TODO we ultimately want to be able to have weight-and-capacity-aware m-n host2component
        # assignments, not just round robin of the whole host2component

    component_i = 0
    for host, workers in migrants.items():
        component_id = components[component_i][1]
        state = timer(migrate_to_component, Microtrace.ctrl_migrate)(host, component_id, state)
        yield from assign_within_component(state, workers, component_id)
        component_i = (component_i + 1) % len(components)

def _set_preparing_at(dataset: DatasetId, worker: WorkerId, state: State, children: list[TaskId]) -> State:
    state.host2ds[worker.host][dataset] = DatasetStatus.preparing
    state.ds2host[dataset][worker.host] = DatasetStatus.preparing
    state.worker2ds[worker][dataset] = DatasetStatus.preparing
    state.ds2worker[dataset][worker] = DatasetStatus.preparing
    # TODO check that there is no invalid transition? Eg, if it already was preparing or available
    # TODO do we want to do anything for the other workers on the same host? Probably not, rather consider
    # host2ds during assignments

    for task in children:
        component = state.ts2component[task]
        state = update_worker2task_distance(component.worker2task_distance, task, worker, state)
        maybe_new_opt = component.worker2task_distance[worker][task]
        if (value := component.computable.get(task, None)) is not None and value > maybe_new_opt:
            component.computable[task] = maybe_new_opt
    return state

def plan(state: State, assignments: list[Assignment]) -> State:
    """Given actions that were just sent to a worker, update state to reflect it, including preparation
    and planning for future assignments.
    Unlike `assign`, this is less performance critical, so slightly longer calculations can happen here.
    """

    # TODO when considering `children` below, filter for near-computable? Ie, either already in computable
    # or all inputs are already in preparing state? May not be worth it tho

    for assignment in assignments:
        for prep in assignment.prep:
            children = state.task_o[prep[0]]
            state = _set_preparing_at(prep[0], assignment.worker, state, children)
        for task in assignment.tasks:
            for ds in assignment.outputs:
                children = state.edge_o[ds]
                state = _set_preparing_at(ds, assignment.worker, state, children)
            state.worker2ts[assignment.worker][task] = TaskStatus.enqueued
            state.ts2worker[task][assignment.worker] = TaskStatus.enqueued
            state.ongoing.add(task)

    return state
