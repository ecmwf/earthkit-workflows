"""
Implementation of controller.api.Executor for Dask using Futures

Caveats:
 - Does not support fine-grained scatter: every output is simply accessible to all after task computing it completes.
   Similarly, `hosts` is ignored in the `purge` call.
 - Does not support fetch_by_url -- raises exception when tempted in such fashion.
"""

# TODO move to cascade.executor, or cascade-dask package

import logging
import uuid
from typing import Any, Callable

from dask.distributed import Client, Future, Variable, get_client, wait
from dask.distributed.deploy.cluster import Cluster

from cascade.controller.api import ExecutableTaskInstance, ExecutableSubgraph
from cascade.low.core import Environment, Host, TaskDefinition, NO_OUTPUT_PLACEHOLDER
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
def _get_output(task: str, output_key: str) -> Variable:
    return Variable(f"{task}-{output_key}")


# wrapper to be the target of the Dask Future
def execute_subgraph(subgraph: ExecutableSubgraph) -> None:
    logger.debug(f"preparing {subgraph=}")
    local_outputs: dict[tuple[str, str], Any] = {} # TODO replace with eg zict to handle overflows
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
            if (w.sourceTask, w.sourceOutput) in local_outputs:
                logger.debug(f"about to get input {w} from in-process mem")
                value = local_outputs[(w.sourceTask, w.sourceOutput)]
            else:
                logger.debug(f"about to get input {w} from dask var")
                value = _get_output(w.sourceTask, w.sourceOutput).get().result()
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

        local_outputs[(task.name, output_key)] = res
        if output_key in task.published_outputs:
            res_var = _get_output(task.name, output_key)
            client = get_client()
            res_fut = client.scatter(res)
            res_var.set(res_fut)


class DaskFuturisticExecutor:
    def __init__(self, cluster: Cluster, environment: Environment|None = None):
        self.cluster = cluster
        self.client = Client(cluster)
        self.futures: dict[str, Future] = {}
        self.task2future: dict[str, str] = {}
        # TODO utilize self.client.benchmark_hardware() or access the worker mem specs
        if not environment:
            environment = Environment(
                hosts={
                    _worker_id_from(w): Host(memory_mb=1, cpu=1, gpu=0)
                    for w in self.cluster.workers
                }
            )
        if set(environment.hosts.keys()) != set(_worker_id_from(w) for w in self.cluster.workers):
            raise ValueError(f"inconsistency between {environment=} and {self.cluster.workers=}")
        self.environment = environment

    def get_environment(self) -> Environment:
        return self.environment

    def run_at(self, subgraph: ExecutableSubgraph, host: str) -> str:
        fut = self.client.submit(execute_subgraph, subgraph, workers=_worker_id_to(host))
        while True:
            # NOTE I originally tried fut._uid, but that appears to be recycled?
            id_ = str(uuid.uuid4())
            if id_ not in self.futures:
                break
        self.futures[id_] = fut
        for task in subgraph.tasks:
            if task.name in self.task2future:
                logger.warning(
                    f"task {task.name} has already had future {self.task2future[task.name]}, assuming retry and overriding"
                )
            self.task2future[task.name] = id_
            logger.debug(f"registering {task.name} under future {id_}")
        return id_

    def scatter(self, taskName: str, outputName: str, hosts: set[str]) -> str:
        # NOTE see the caveat at the top -- we don't finegrain
        if taskName not in self.task2future:
            raise ValueError(
                f"task computing this output hasn't been submitted: {taskName}, {outputName}"
            )
        return self.task2future[taskName]

    def purge(
        self, taskName: str, outputName: str, hosts: set[str] | None = None
    ) -> None:
        if hosts:
            logger.warning("param host given, but ignored")
        var = _get_output(taskName, outputName)
        var.delete()

    def fetch_as_url(self, taskName: str, outputName: str) -> str:
        raise NotImplementedError

    def fetch_as_value(self, taskName: str, outputName: str) -> Any:
        return _get_output(taskName, outputName).get().result()

    def wait_some(self, ids: set[str], timeout_sec: int | None = None) -> set[str]:
        futs = [self.futures[e] for e in ids]
        finished = set(
            e._uid
            for e in wait(futs, return_when="FIRST_COMPLETED", timeout=timeout_sec).done
        )
        return finished

    def is_done(self, id_: str, timeout: int | None = None) -> bool:
        logger.debug(f"inquiring completion of {id_}")
        if self.futures[id_].status == 'error':
            logger.error(f"future {id_} failed")
            raise ValueError(id_) # TODO better
        return self.futures[id_].done()
