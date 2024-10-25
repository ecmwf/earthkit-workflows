"""
Implementation of controller.api.Executor for Dask using Futures

Caveats:
 - Does not support fine-grained scatter: every output is simply accessible to all after task computing it completes.
   Similarly, `hosts` is ignored in the `purge` call.
 - Does not support fetch_by_url -- raises exception when tempted in such fashion.
"""

import logging
import uuid
from typing import Any, Callable, Iterable
from dataclasses import dataclass

from dask.distributed import Client, Future, Variable, get_client, wait
from dask.distributed.deploy.cluster import Cluster

from cascade.low.views import param_source
from cascade.controller.core import DatasetStatus, TaskStatus, Event, ActionDatasetTransmit, ActionSubmit, ActionDatasetPurge, TaskInstance
from cascade.executors.instant import SimpleEventQueue
from cascade.low.core import Environment, Worker, TaskDefinition, NO_OUTPUT_PLACEHOLDER, DatasetId, TaskId, WorkerId, JobInstance
from cascade.low.func import ensure, maybe_head

logger = logging.getLogger(__name__)


# dask quirk section
def _worker_id_from(v: Any) -> str:
    if isinstance(v, int):
        return str(v)
    elif isinstance(v, str):
        return v
    else:
        raise TypeError(f"impossible conversion: {v} of {type(v)}")


def _worker_id_to(v: str) -> str | int:
    try:
        return int(v)
    except ValueError:
        return v


# we rely on variables for cross task comms
def _get_output(dataset_id: DatasetId) -> Variable:
    return Variable(f"{dataset_id.task}-{dataset_id.output}")

@dataclass
class VariableWiring:
    """A view of source of an edge, ie, how a TaskInstance obtains a dynamic input"""

    source: DatasetId
    intoKwarg: str|None
    intoPosition: int|None
    annotation: str

@dataclass
class ExecutableTaskInstance:
    """A wrapper around TaskInstance that contains necessary means of execution"""

    task: TaskInstance
    name: str
    wirings: list[VariableWiring]

@dataclass
class ExecutableSubgraph:
    tasks: list[ExecutableTaskInstance]
    published_outputs: set[DatasetId]

# wrapper to be the target of the Dask Future
def execute_subgraph(subgraph: ExecutableSubgraph) -> None:
    logger.debug(f"preparing {subgraph=}")
    local_outputs: dict[DatasetId, Any] = {} # TODO replace with eg zict to handle overflows
    for task in subgraph.tasks:
        func: Callable
        if task.task.definition.func is not None:
            func = TaskDefinition.func_dec(task.task.definition.func)
        else:
            raise NotImplementedError
            # TODO just importlib parse task.task.definition.entrypoint
        kwargs = task.task.static_input_kw.copy()
        args: list[Any] = []
        for i, a in task.task.static_input_ps.items():
            ensure(args, i)
            args[i] = a

        for w in task.wirings:
            if w.source in local_outputs:
                logger.debug(f"about to get input {w} from in-process mem")
                value = local_outputs[w.source]
            else:
                logger.debug(f"about to get input {w} from dask var")
                value = _get_output(w.source).get().result()
            logger.debug(f"input {w} has value {value}")
            if w.intoKwarg is not None:
                kwargs[w.intoKwarg] = value
            if w.intoPosition is not None:
                ensure(args, w.intoPosition)
                args[w.intoPosition] = value

        if len(task.task.definition.output_schema) > 1:
            raise NotImplementedError
        output_key = maybe_head(task.task.definition.output_schema.keys())
        if not output_key:
            raise ValueError(f"no output key for task {task.name}!")

        logger.debug("executing func")
        try:
            res = func(*args, **kwargs)
        except Exception:
            logger.exception(f"{task.name=} failed in execution")
            raise
        if output_key == NO_OUTPUT_PLACEHOLDER:
            if res is not None:
                logger.warning(f"gotten output of type {type(res)} where none was expected!")
            else:
                res = "ok"

        # TODO purge local outputs that won't be needed later in the process

        output_id = DatasetId(task.name, output_key)
        local_outputs[output_id] = res
        if output_id in subgraph.published_outputs:
            logger.debug(f"scattering for {output_id}")
            res_var = _get_output(output_id)
            client = get_client()
            res_fut = client.scatter(res)
            res_var.set(res_fut)
            logger.debug(f"result {output_id} represented with {res_fut}")


class DaskFuturisticExecutor:
    def __init__(self, cluster: Cluster, job: JobInstance, environment: Environment|None = None):
        self.cluster = cluster
        self.client = Client(cluster)
        self.fid2action: dict[str, ActionSubmit] = {}
        self.fid2future: dict[str, Future] = {}
        self.eq = SimpleEventQueue()
        # TODO utilize self.client.benchmark_hardware() or access the worker mem specs
        if not environment:
            environment = Environment(
                workers={
                    _worker_id_from(w): Worker(memory_mb=1, cpu=1, gpu=0)
                    for w in self.cluster.workers
                }
            )
        if set(environment.workers.keys()) != set(_worker_id_from(w) for w in self.cluster.workers):
            raise ValueError(f"inconsistency between {environment=} and {self.cluster.workers=}")
        self.environment = environment
        self.job = job
        self.param_source = param_source(job.edges)

    def _build_executable(self, action: ActionSubmit) -> ExecutableSubgraph:
        tasks = [
            ExecutableTaskInstance(
                name=task,
                task=self.job.tasks[task],
                wirings=[
                    VariableWiring(
                        source=dataset_id,
                        intoKwarg=k if isinstance(k, str) else None,
                        intoPosition=k if isinstance(k, int) else None,
                        annotation=self.job.tasks[dataset_id.task].definition.output_schema[dataset_id.output],
                    )
                    for k, dataset_id in self.param_source[task].items()
                ]
            )
            for task in action.tasks
        ]
        return ExecutableSubgraph(tasks=tasks, published_outputs=action.outputs)


    def get_environment(self) -> Environment:
        return self.environment

    def submit(self, action: ActionSubmit) -> None:
        fut = self.client.submit(execute_subgraph, self._build_executable(action), workers=_worker_id_to(action.at))
        self.fid2action[fut._uid] = action
        self.fid2future[fut._uid] = fut

    def transmit(self, action: ActionDatasetTransmit) -> None:
        logger.warning("ignoring transmit due to everything-broadcasted")
        # NOTE see the caveat at the top -- we don't finegrain
        self.eq.transmit_done(action)
    
    def purge(self, action: ActionDatasetPurge) -> None:
        logger.warning("ignoring purge's host spec {action.at}")
        # NOTE see the caveat at the top -- we don't finegrain
        for ds in action.ds:
            var = _get_output(ds)
            var.delete()

    def fetch_as_url(self, dataset_id: DatasetId) -> str:
        raise NotImplementedError

    def fetch_as_value(self, dataset_id: DatasetId) -> Any:
        return _get_output(dataset_id).get().result()

    def wait_some(self, timeout_sec: int | None = None) -> Iterable[Event]:
        if self.eq.any():
            return self.eq.drain()

        running = list(self.fid2future.values())
        logger.debug(f"awaiting on {running}")
        result = wait(running, return_when="FIRST_COMPLETED", timeout=timeout_sec)
        logger.debug(f"awaited futures {result}")
        for fut in result.done:
            if fut.status == 'error':
                logger.error(f"future {fut} corresponding to {self.fid2action[fut._uid]} failed")
                raise ValueError(fut)
            self.eq.submit_done(self.fid2action.pop(fut._uid))
            self.fid2future.pop(fut._uid)

        return self.eq.drain()
