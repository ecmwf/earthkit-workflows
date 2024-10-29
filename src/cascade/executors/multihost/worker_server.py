"""
Simple facade in front of (any) `controller.executor` implementation. Works in the same
process/thread as the executor itself -- presumably suboptimal wrt the data transmit calls.

Most calls are trivially forwarded to the underlying executor, except for `transmit` which
needs to initiate its own connection, and `wait_some` which needs to introduce an async
wrapper.
"""

from pydantic import BaseModel
from cascade.low.core import DatasetId
from cascade.controller.executor import Executor
from cascade.controller.core import ActionSubmit, ActionDatasetPurge, ActionDatasetTransmit
from starlette.responses import Response, JSONResponse, StreamingResponse, PlainTextResponse
from starlette.requests import Request
from starlette.applications import Starlette
from starlette.routing import Route
import logging
import orjson
from typing import AsyncIterator, TypedDict
from contextlib import asynccontextmanager
import httpx

logger = logging.getLogger(__name__)

class TransmitPayload(BaseModel):
    # corresponds to ActionDatasetTransmit -- but since it happens across workers, we cant
    # just reuse the original model
    other_url: str
    other_worker: str
    this_worker: str
    datasets: list[DatasetId]

class OrjsonResponse(JSONResponse):
    def render(self, content: dict|list) -> bytes:
        return orjson.dumps(content)

ok_response = Response()

class State(TypedDict):
    executor: Executor
    client: httpx.AsyncClient

async def status(request: Request) -> Response:
    print("gotten status")
    return ok_response

# get, () -> (Environment)
async def get_environment(request: Request) -> Response:
    env = request.state.executor.get_environment()
    return OrjsonResponse(env.model_dump())

# put, (ActionSubmit) -> ()
async def submit(request: Request) -> Response:
    action = ActionSubmit(**(await request.json()))
    logger.debug(f"recieved submit {action=}")
    request.state.executor.submit(action)
    return ok_response

# put, ({worker}/{dataset_id}) -> ()
async def store_value(request: Request) -> Response:
    worker: str = request.path_params['worker']
    task: str = request.path_params['dataset_task']
    output: str = request.path_params['dataset_output']
    dataset = DatasetId(task, output)
    data = await request.body()
    request.state.executor.store_value(worker, dataset, data)
    return ok_response

# post, (TransmitPayload) -> ()
async def transmit_remote(request: Request) -> Response:
    # TODO async read from executor, async stream to other party? Or submit whole thing to thread pool?
    payload = TransmitPayload(**(await request.json()))
    executor = request.state.executor
    client = request.state.client
    url_base = f"{payload.other_url}/store_value/{payload.other_worker}"
    for dataset in payload.datasets:
        logger.debug(f"fetching {dataset=} for transmit")
        data = executor.fetch_as_value(payload.this_worker, dataset)
        url = f"{url_base}/{dataset.task}/{dataset.output}"
        logger.debug(f"transmitting {dataset=}")
        rv = await client.put(url, content=bytes(data.view()))
        data.close()
    return ok_response

# post, (ActionDatasetTransmit) -> ()
async def transmit_local(request: Request) -> Response:
    action = ActionDatasetTransmit(**(await request.json()))
    request.state.executor.transmit(action)
    return ok_response

# post, (ActionDatasetPurge) -> ()
async def purge(request: Request) -> Response: 
    action = ActionDatasetPurge(**(await request.json()))
    request.state.executor.purge(action)
    return ok_response

# get, ({worker}/{dataset_id}) -> (str)
async def fetch_as_url(request: Request) -> Response:
    worker: str = request.path_params['worker']
    task: str = request.path_params['dataset_task']
    output: str = request.path_params['dataset_output']
    dataset = DatasetId(task, output)
    url = request.state.executor.fetch_as_url(worker, dataset)
    return PlainTextResponse(url)

# get, ({worker}/{dataset_id}) -> (Any)
async def fetch_as_value(request: Request) -> Response:
    worker: str = request.path_params['worker']
    task: str = request.path_params['dataset_task']
    output: str = request.path_params['dataset_output']
    dataset = DatasetId(task, output)
    value = request.state.executor.fetch_as_value(worker, dataset)
    if not isinstance(value, bytes):
        raise TypeError("expected bytes in {dataset=}, gotten {type(value)} instead")
    return Response(value, media_type='application/octet-stream')

# get, () -> (list[Events])
async def wait_some(request: Request) -> Response:
    executor = request.state.executor
    events = executor.wait_some() # TODO add somehow await here
    logger.debug(f"reporting {events=}")
    return OrjsonResponse([e.model_dump() for e in events])

def build_app(executor: Executor, is_debug: bool = False):

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[State]:
        client = httpx.AsyncClient()
        yield {"executor": executor, "client": client }
        await client.aclose()

    return Starlette(
        debug=is_debug,
        routes = [
            Route('/status', status, methods=["GET", "HEAD"]),
            Route('/get_environment', get_environment, methods=["GET"]),
            Route('/submit', submit, methods=["PUT"]),
            Route('/transmit_remote', transmit_remote, methods=["POST"]),
            Route('/transmit_local', transmit_local, methods=["POST"]),
            Route('/purge', purge, methods=["POST"]),
            Route('/fetch_as_url/{worker}/{dataset_task}/{dataset_output}', fetch_as_url, methods=["GET"]),
            Route('/fetch_as_value/{worker}/{dataset_task}/{dataset_output}', fetch_as_value, methods=["GET"]),
            Route('/store_value/{worker}/{dataset_task}/{dataset_output}', store_value, methods=["PUT"]),
            Route('/wait_some', wait_some, methods=["GET"]),
        ],
        lifespan=lifespan,
    )
