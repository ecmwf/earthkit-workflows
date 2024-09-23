"""
Adapter of the core graph definition into execution via Dask Futures
"""

# TODO major improvements:
# - delete variables early, once all consumers finished
# - scatter variables to the workers that will need them
# - support for multiple outputs / generators per task
# - control precisely which futures to launch already -- currently we just throw in first 2 in the queue per host
# - handle failed states, restarts, etc

from cascade.v2.func import ensure, maybe_head
from dask.distributed import Variable, wait, Client, Future, get_client
from dataclasses import dataclass
from typing import Optional, Callable, Any
from cascade.v2.core import JobInstance, TaskDefinition, TaskInstance, Schedule
from cascade.v2.views import param_source
import logging

logger = logging.getLogger(__name__)

@dataclass
class VariableWiring:
    varName: str
    intoKwarg: Optional[str]
    intoPosition: Optional[int]
    

@dataclass
class FuturePayload:
    task: TaskInstance
    name: str
    wirings: list[VariableWiring]

def get_var_name(task: str, output_key: str) -> str:
    return f"{task}-{output_key}"

def execute_future_payload(payload: FuturePayload) -> None:
    logger.debug(f"preparing {payload=}")
    func: Callable
    if payload.task.definition.func is not None:
        func = TaskDefinition.func_dec(payload.task.definition.func)
    else:
        raise NotImplementedError
        # TODO just importlib parse payload.task.definition.entrypoint
    kwargs = payload.task.static_input_kw.copy()
    args: list[Any] = []
    for i, a in payload.task.static_input_ps.items():
        ensure(args, i)
        args[i] = a

    for w in payload.wirings:
        logger.debug(f"about to get input {w.varName}")
        value = Variable(w.varName).get().result()
        if w.intoKwarg is not None:
            kwargs[w.intoKwarg] = value
        if w.intoPosition is not None:
            ensure(args, w.intoPosition) 
            args[w.intoPosition] = value

    logger.debug(f"executing func")
    res = func(*args, **kwargs)
    if len(payload.task.definition.output_schema) > 1:
        raise NotImplementedError
    output_key = maybe_head(payload.task.definition.output_schema.keys())
    if not output_key:
        raise ValueError
    res_var_name = get_var_name(payload.name, output_key)
    logger.debug(f"about to propagate result {res_var_name}")
    res_var = Variable(res_var_name)
    client = get_client()
    res_fut = client.scatter(res)
    res_var.set(res_fut)

def task2future(
    name: str, task: TaskInstance, input2source: dict[int | str, tuple[str, str]]
) -> FuturePayload:
    return FuturePayload(
        task=task, 
        name=name,
        wirings=[
            VariableWiring(
                varName=get_var_name(v[0], v[1]),
                intoKwarg = k if isinstance(k, str) else None,
                intoPosition = k if isinstance(k, int) else None,
            )
            for k, v in input2source.items()
        ]
    )

def execute_via_futures(job: JobInstance, schedule: Schedule, final_output: tuple[str, str], client: Client) -> Any:
    task_param_sources = param_source(job.edges)
    future_payloads = {
        task: task2future(task, instance, task_param_sources[task])
        for task, instance in job.tasks.items()
    }
    
    host_ongoing: dict[str|int, list[Future]] = {host: [] for host in schedule.host_task_queues.keys()}
    host_remaining = {host: tasks for host, tasks in schedule.host_task_queues.items()}
    while True:
        for host in schedule.host_task_queues.keys():
            while len(host_ongoing[host]) > 0 and host_ongoing[host][0].done():
                completed = host_ongoing[host].pop(0)
                logger.debug(f"finished {completed} on {host}")
            while len(host_ongoing[host]) < 2 and len(host_remaining[host]) > 0:
                nextFutPayload = future_payloads[host_remaining[host].pop(0)]
                logger.debug(f"submitting {nextFutPayload} on workers {host}")
                host_ongoing[host].append(client.submit(execute_future_payload, nextFutPayload, workers=host))
        ongoing = [fut for futs in host_ongoing.values() for fut in futs]
        if len(ongoing) > 0:
            logger.debug(f"awaiting on {len(ongoing)} futures")
            wait(ongoing, return_when='FIRST_COMPLETED')
        else:
            logger.debug(f"nothing runnin, breaking")
            break


    result = Variable(get_var_name(final_output[0], final_output[1])).get().result()
    for name, instance in job.tasks.items():
        for key in instance.definition.output_schema.keys():
            var_name = get_var_name(name, key)
            var = Variable(var_name)
            logger.debug(f"deleting var {var_name=} {var.name=}")
            var.delete()
    return result
