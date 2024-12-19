from cascade.scheduler.core import State, Preschedule, ComponentSchedule, ComponentId
from cascade.low.core import Environment, WorkerId, DatasetId, JobInstance
from cascade.scheduler.assign import assign_within_component, migrate_to_component

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
    for (componentId, precomponent), host in zip(enumerate(preschedule.components), host2workers):
        # TODO improve this naive -- round robin, does allow neither m-to-1 nor 1-to-m for host2component
        component = ComponentSchedule(
            core=precomponent,
            computable={task: 0 for task in precomponent.sources},
            worker2task_distance={
                worker: {task: 0 for task in precomponent.sources} for worker in host2workers[host]
            },
        )
        host2component[host] = componentId
        components.append(component)
        computable += len(precomponent.sources)

    return State(
        edge_o=preschedule.edge_o,
        edge_i=preschedule.edge_i,
        components=components,
        host2component=host2component,
        purging_tracker=purging_tracker,
        outputs={e: None for e in outputs},
        worker_colocations={worker: set(host2workers[worker]) for worker in environment.workers},
        computable=computable,
    )

def assign(state: State, component: int|None = None) -> Iterator[Assignment]:
    """Given idle workers in `state`, assign actions to workers. Mutates the state:
     - pops from computable & idle workers,
     - decreases weight,
     - changes host2component.
    Yields, to allow for immediate async sending to workers.
    Performance critical section, we need to output an assignment asap. Steps taking longer
    should be deferred to `plan`"""

    # TODO use this somewhere
    # timer(callable, Microtrace.ctrl_assign)

    # step I: assign within existing components
    components = defaultdict(list)
    for worker in workers:
        components[worker.host].append(worker)

    for component_id, local_workers in components.items():
        yield from assign_within_component(state, local_workers, component_id)
    
    if remaining_w and remaining_t:
        raise ValueError(f"unexpected state: {remaining_t=} but {remaining_w=}")
    if not remaining_w:
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

    component_i = 0
    for host, workers in migrants.items():
        component_id = components[component_i][1]
        state = migrate_to_component(host, component_id, state)
        yield from assign_within_component(state, workers, component_id)
        component_i = (component_i + 1) % len(components)

def plan(state: State, assignments: list[Assignment]) -> State:
    """Given actions that were just sent to a worker, update state to reflect it, including preparation
    and planning for future assignments.
    Unlike `assign`, this is less performance critical, so slightly longer calculations can happen here.
    """
    # needs to change:
    # ds2worker, ts2worker, ds2host to reflect planning/submission
    # component.computable to have newly made available tasks -- beware the planned/available trap!
    # component.worker2task distance to reflect newly present datasets -- planned state is sufficient
    # component.computable to reflect new optimum
    # computable to reflect total state
    # worker2taskOverhead to reflect new costs

    update_dist4new_computable(worker2task: Worker2TaskDistance, task: TaskId, component: ComponentCore, worker2ds: dict[WorkerId, dict[DatasetId, DatasetStatus]])

    raise NotImplementedError
