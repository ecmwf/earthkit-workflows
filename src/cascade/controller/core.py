from dataclasses import dataclass
from cascade.low.core import DatasetId, HostId, TaskId
from typing import Any

@dataclass
class State:
    # key add by core.initialize, value add by notify.notify
    outputs: dict[DatasetId, Any]
    # add by notify.consider_fetch, remove by act.flush_queues
    fetching_queue: dict[DatasetId, HostId]
    # add by notify.consider_purge, removed by act.flush_queues
    # TODO extend with `at`, for fine graining?
    purging_queue: list[DatasetId]
    # add by core.initialize, remove by notify.notify
    purging_tracker: dict[DatasetId, set[TaskId]]

    def has_awaitable(self) -> bool:
        # TODO replace the None in outputs with check on fetch queue (but change that from binary to ternary first)
        return None in self.outputs.values()


def init_state(outputs: set[DatasetId], edge_o: dict[DatasetId, set[TaskId]]) -> State:
    purging_tracker = {
        ds: {task for task in dependants}
        for ds, dependants in edge_o.items()
    }

    return State(
        outputs={e: None for e in outputs},
        fetching_queue={},
        purging_queue=[],
        purging_tracker=purging_tracker
    )
