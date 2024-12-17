"""
Adapter between a backbone and an instance of Executor that runs on the host
"""

import logging
import base64
from cascade.controller.executor import Executor
from cascade.controller.core import ActionSubmit, ActionDatasetTransmit, ActionDatasetPurge, DatasetId, Event, WorkerId, TransmitPayload, DatasetStatus
from cascade.executors.backbone.interface import Backbone
from cascade.executors.backbone.serde import RegisterRequest, RegisterResponse, Shutdown, DataTransmitObject, DatasetFetch
from cascade.low.func import assert_never
from cascade.shm.client import AllocatedBuffer
from cascade.controller.tracing import mark, TransmitLifecycle

logger = logging.getLogger(__name__)

class BackboneLocalExecutor():
    def __init__(self, executor: Executor, backbone: Backbone):
        self.executor = executor
        self.backbone = backbone
        self.executor.register_event_callback(self.backbone.send_event_callback())
        self.shutdown = False


    def _send_data(self, payload: TransmitPayload) -> None:
        callback = self.backbone.send_event_callback()
        for dataset in payload.datasets:
            try:
                mark({"dataset": dataset.task, "action": TransmitLifecycle.started, "source": payload.this_worker, "target": payload.other_worker, "host": payload.tracing_ctx_host, "mode": "remote"})
                data = self.executor.fetch_as_value(payload.this_worker, dataset)
                mark({"dataset": dataset.task, "action": TransmitLifecycle.loaded, "source": payload.this_worker, "target": payload.other_worker, "host": payload.tracing_ctx_host, "mode": "remote"})
                if isinstance(data, AllocatedBuffer):
                    data_raw = bytes(data.view()) # NOTE unfortunate -- zmq supports going with view, but that needs a serde extension
                else:
                    data_raw = data
                dto = DataTransmitObject(
                    worker_id=payload.other_worker,
                    dataset_id=dataset,
                    data=data_raw,
                )
                self.backbone.send_data(payload.other_url, dto)
                if isinstance(data, AllocatedBuffer):
                    data.close()
                # event = Event(at=payload.other_worker, ts_trans=[], ds_trans=[(dataset, DatasetStatus.available)])
            except Exception as ex:
                # NOTE in case of success, we rely on the other party to submit
                event = Event(failures=[f"data transmit of {dataset} failed with {repr(ex)}"], at=payload.this_worker)
                callback(event)
                logger.debug(f"callback of {event} finished")

    def recv_loop(self) -> None:
        while not self.shutdown:
            for m in self.backbone.recv_messages():
                if isinstance(m, ActionSubmit):
                    self.executor.submit(m)
                elif isinstance(m, ActionDatasetPurge):
                    self.executor.purge(m)
                elif isinstance(m, ActionDatasetTransmit):
                    raise TypeError
                elif isinstance(m, TransmitPayload):
                    self._send_data(m)
                elif isinstance(m, Event):
                    # TODO this is a very poor hack. We just need to ensure the right p.join happen
                    # TODO we are fail to react when no Event comes due to proc crash. It sounds like the executor
                    # should have a thread to check states of processes, and send Failure events in case
                    if hasattr(self.executor, "procwatch"):
                        tasks = {ts[0] for ts in m.ts_trans}
                        logger.debug(f"noted finish of {tasks=}")
                        futs = {k for k, v in self.executor.fid2action.items() if set(v.tasks).intersection(tasks)}
                        logger.debug(f"will proc join {futs=}")
                        self.executor.procwatch.spawn_available(should_wait=futs)
                elif isinstance(m, RegisterRequest):
                    raise TypeError
                elif isinstance(m, RegisterResponse):
                    raise TypeError
                elif isinstance(m, Shutdown):
                    logger.debug("shutting down local executor")
                    self.executor.shutdown()
                    logger.debug("shutting down local backbone")
                    self.backbone.shutdown()
                    logger.debug("exiting recv loop")
                    self.shutdown = True
                elif isinstance(m, DataTransmitObject):
                    self.executor.store_value(m.worker_id, m.dataset_id, m.data)
                elif isinstance(m, DatasetFetch):
                    data = self.executor.fetch_as_value(m.worker, m.dataset)
                    if isinstance(data, AllocatedBuffer):
                        data_raw = bytes(data.view()) # NOTE unfortunate -- zmq supports going with view, but that needs a serde extension
                    else:
                        data_raw = data
                    dto = Event(at=m.worker, ds_fetch=[(m.dataset, base64.b64encode(data_raw))])
                    logger.debug(f"sending fetch of {m.dataset=}")
                    self.backbone.send_message('controller', dto)
                    if isinstance(data, AllocatedBuffer):
                        data.close()
                else:
                    assert_never(m)

