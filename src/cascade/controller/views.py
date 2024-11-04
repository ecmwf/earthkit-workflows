from cascade.low.core import Environment, WorkerId, DatasetId
import logging
from cascade.controller.core import State, DatasetStatus
from typing import Generator

logger = logging.getLogger(__name__)

def colocated_workers(env: Environment) -> dict[WorkerId, set[WorkerId]]:
    d = {
        worker: {worker}
        for worker in env.workers
    }
    for cluster in env.colocations:
        for worker in cluster:
            d[worker] = d[worker].union(cluster)
    return d

def project_colocation(workers: list[WorkerId], colocated: dict[WorkerId, set[WorkerId]]) -> set[WorkerId]:
    """Filter list of workers such that no two from the same colocation are chosen"""
    chosen: set[WorkerId] = set()
    for worker in workers:
        if chosen.intersection(colocated[worker]):
            continue
        chosen.add(worker)
    return chosen

def transition_dataset(state: State, source: WorkerId, dataset: DatasetId, to: DatasetStatus) -> State:
    """Ensures valid update of State internal structures, including dataset broadcast in colocated workers"""
    # NOTE currently the implementation is mutating, but we may replace with pyrsistent etc.
    # Thus the caller always *must* use the return value and cease using the input.
    # NOTE not really a `view`, but used in both act & notify, so putting here
    if to == DatasetStatus.missing:
        raise ValueError(f"invalid transition {to = }")
    for worker in state.worker_colocations[source]:
        logger.debug(f"transitioning {dataset} at {worker} to {to}")
        orig_state1 = state.worker2ds[worker].get(dataset, DatasetStatus.missing)
        orig_state2 = state.ds2worker[dataset].get(worker, DatasetStatus.missing)
        if orig_state2 != orig_state1:
            raise ValueError(f"state inconsistency: {dataset=} {worker=} {orig_state1=} {orig_state2=}")
        if orig_state1.value > to.value:
            raise ValueError(f"invalid transition: {dataset=} {worker=} {orig_state1=} {to=}")
        if to == DatasetStatus.purged:
            state.worker2ds[worker].pop(dataset)
            state.ds2worker[dataset].pop(worker)
        else:
            state.worker2ds[worker][dataset] = to
            state.ds2worker[dataset][worker] = to
    return state
