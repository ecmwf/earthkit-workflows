from cascade.low.core import Environment, WorkerId, DatasetId
import logging
from cascade.controller.core import State, DatasetStatus
from typing import Generator

logger = logging.getLogger(__name__)

def project_colocation(workers: list[WorkerId], colocated: dict[WorkerId, set[WorkerId]]) -> set[WorkerId]:
    """Filter list of workers such that no two from the same colocation are chosen"""
    chosen: set[WorkerId] = set()
    for worker in workers:
        if chosen.intersection(colocated[worker]):
            continue
        chosen.add(worker)
    return chosen

def get_host2datasets(state: State, environment: Environment) -> dict[str, set[DatasetId]]:
    # NOTE there is unfortunate mind split when it comes to "does worker identity matter"?
    # For current workers, not because the processes are short lived. But if that ceases to be
    # then this method makes less sense. If it would never matter, then this method should instead
    # become State's field
    return {
        colocation[0].split(':', 1)[0]: {
            dataset
            for dataset, status in state.worker2ds[colocation[0]].items()
            if status == DatasetStatus.available
        }
        for colocation in environment.colocations
    }

def get_some_worker_with(state: State, dataset: DatasetId) -> WorkerId|None:
    for worker, status in state.ds2worker[dataset].items():
        if status == DatasetStatus.available:
            return worker
    return None
