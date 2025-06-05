"""Declares core data structures and methods -- most notably Controller's state,
which handles dynamic work outside of scheduler control, such as dataset purging
or outputs retrieval
"""

import logging
from dataclasses import dataclass
from typing import Any, Iterator

import cascade.executor.serde as serde
from cascade.executor.msg import DatasetTransmitPayload
from cascade.low.core import DatasetId, HostId, TaskId

logger = logging.getLogger(__name__)


@dataclass
class State:
    # key add by core.initialize, value add by notify.notify
    outputs: dict[DatasetId, Any]
    # add by notify.notify, remove by act.flush_queues
    fetching_queue: dict[DatasetId, HostId]
    # add by notify.notify, removed by act.flush_queues
    purging_queue: list[DatasetId]
    # add by core.init_state, remove by notify.notify
    purging_tracker: dict[DatasetId, set[TaskId]]

    def has_awaitable(self) -> bool:
        # TODO replace the None in outputs with check on fetch queue (but change that from binary to ternary first)
        # NOTE this `return None in self.outputs.values()` doesnt work because of numpy `truth value ambiguous`
        for e in self.outputs.values():
            if e is None:
                return True
        return False

    def _consider_purge(self, dataset: DatasetId) -> None:
        """If dataset not required anymore, add to purging_queue"""
        no_dependants = not self.purging_tracker.get(dataset, None)
        not_required_output = self.outputs.get(dataset, 1) is not None
        if no_dependants and not_required_output:
            logger.debug(f"adding {dataset=} to purging queue")
            if dataset in self.purging_tracker:
                self.purging_tracker.pop(dataset)
            self.purging_queue.append(dataset)

    def consider_fetch(self, dataset: DatasetId, at: HostId) -> None:
        """If required as output and not yet arrived, add to fetching queue"""
        if (
            dataset in self.outputs
            and self.outputs[dataset] is None
            and dataset not in self.fetching_queue
        ):
            self.fetching_queue[dataset] = at

    def receive_payload(self, payload: DatasetTransmitPayload) -> None:
        """Stores deserialized value into outputs, considers purge"""
        # NOTE ifneedbe get annotation from job.tasks[event.ds.task].definition.output_schema[event.ds.output]
        self.outputs[payload.header.ds] = serde.des_output(
            payload.value, "Any", payload.header.deser_fun
        )
        self._consider_purge(payload.header.ds)

    def task_done(self, task: TaskId, inputs: set[DatasetId]) -> None:
        """Marks that the inputs are not needed for this task anymore, considers purge of each"""
        for sourceDataset in inputs:
            self.purging_tracker[sourceDataset].remove(task)
            self._consider_purge(sourceDataset)

    def drain_purging_queue(self) -> Iterator[DatasetId]:
        for e in self.purging_queue:
            yield e
        self.purging_queue = []

    def drain_fetching_queue(self) -> Iterator[tuple[DatasetId, HostId]]:
        for dataset, host in self.fetching_queue.items():
            yield dataset, host
        self.fetching_queue = {}


def init_state(outputs: set[DatasetId], edge_o: dict[DatasetId, set[TaskId]]) -> State:
    purging_tracker = {
        ds: {task for task in dependants} for ds, dependants in edge_o.items()
    }

    return State(
        outputs={e: None for e in outputs},
        fetching_queue={},
        purging_queue=[],
        purging_tracker=purging_tracker,
    )
