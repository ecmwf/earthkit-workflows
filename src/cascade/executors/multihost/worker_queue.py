"""
Handles tasks executed in their own threads to not block the main async loop.

In particular:
  - remote data transfers
  - awaits for the `executor.wait_some` + data transfer completions
"""

from asyncio import Future, get_running_loop, gather
from cascade.controller.executor import Executor
import httpx
from concurrent.futures import ThreadPoolExecutor
from cascade.controller.core import Event, TransmitPayload, DatasetStatus
from cascade.controller.tracing import mark, TransmitLifecycle
import logging

logger = logging.getLogger(__name__)

# TODO consider moving all executor interactions here and protecting executor with a lock

class WorkerQueue():
    def __init__(self, executor: Executor) -> None:
        self.executor = executor
        self.client = httpx.Client()
        self.futures: list[Future] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=2)

    def submit_transmit(self, payload: TransmitPayload) -> None:
        loop = get_running_loop()
        self.futures.append(loop.run_in_executor(self.thread_pool, self.execute_transmit, payload))

    def execute_transmit(self, payload: TransmitPayload) -> list[Event]:
        rv: list[Event] = []
        try:
            executor = self.executor
            client = self.client
            url_base = f"{payload.other_url}/store_value/{payload.other_worker}"
            for dataset in payload.datasets:
                mark({"dataset": dataset.task, "action": TransmitLifecycle.started, "worker": payload.other_worker, "host": payload.tracing_ctx_host, "mode": "remote"})
                logger.debug(f"fetching {dataset=} for transmit")
                data = executor.fetch_as_value(payload.this_worker, dataset)
                url = f"{url_base}/{dataset.task}/{dataset.output}"
                mark({"dataset": dataset.task, "action": TransmitLifecycle.loaded, "worker": payload.other_worker, "host": payload.tracing_ctx_host, "mode": "remote"})
                logger.debug(f"transmitting {dataset=}")
                callResult = client.put(url, content=bytes(data.view()))
                logger.debug(f"transmit of {dataset=} finished with {callResult}")
                if callResult.status_code != 200:
                    raise ValueError(callResult)
                data.close()
                rv.append(Event(at=payload.other_worker, ts_trans=[], ds_trans=[(dataset, DatasetStatus.available)]))
        except Exception as ex:
            logger.error(f"failed with {repr(ex)} during transmit of {payload}")
            # TODO mark tracing, rich failure report so that worker can retry
            rv.append(Event(failures=[f"data transmit failed with {repr(ex)}"], at=payload.this_worker))
        return rv

       
    async def wait_some(self) -> list[Event]:
        transmits = self.futures
        self.futures = []

        loop = get_running_loop()
        job: Future = loop.run_in_executor(self.thread_pool, self.executor.wait_some)

        logger.debug(f"will await {len(transmits)} ongoing transmits + executor.wait_some")
        # TODO why does gather(job, *transmits) raise RuntimeError await wasnt used with future?
        return [
            event
            for awaitable in [job]+transmits
            for event in (await awaitable)
        ]

    def shutdown(self):
        self.thread_pool.shutdown(wait=True, cancel_futures=True)
        self.client.close()
