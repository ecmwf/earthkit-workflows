"""
Implements the canonical cascade controller
"""

# TODO major improvements
# - scatter variables to the workers that will need them
# - support for multiple outputs / generators per task
# - control precisely which futures to launch already -- currently we just throw in first 2 in the queue per host
# - handle failed states, restarts, etc

import logging

from cascade.controller.api import (
    ExecutableTaskInstance,
    Executor,
    PurgingPolicy,
    VariableWiring,
)
from cascade.low.core import JobInstance, Schedule, TaskInstance
from cascade.low.views import dependants, param_source

logger = logging.getLogger(__name__)


def _task2executable(
    name: str,
    task: TaskInstance,
    job: JobInstance,
    input2source: dict[int | str, tuple[str, str]],
) -> ExecutableTaskInstance:
    return ExecutableTaskInstance(
        task=task,
        name=name,
        wirings=[
            VariableWiring(
                sourceTask=v[0],
                sourceOutput=v[1],
                intoKwarg=k if isinstance(k, str) else None,
                intoPosition=k if isinstance(k, int) else None,
                annotation=job.tasks[v[0]].definition.output_schema[v[1]],
            )
            for k, v in input2source.items()
        ],
    )


def _submit(
    job: JobInstance,
    schedule: Schedule,
    executor: Executor,
    purging_policy: PurgingPolicy,
) -> None:
    task_param_sources = param_source(job.edges)
    executable_tasks = {
        task: _task2executable(task, instance, job, task_param_sources[task])
        for task, instance in job.tasks.items()
    }
    id2task: dict[str, str] = {}

    output_dependants: dict[tuple[str, str], set[str]] = {}
    if purging_policy.eager:
        output_dependants = {
            k: v
            for k, v in dependants(job.edges).items()
            if v and k not in purging_policy.preserve
        }
    host_ongoing: dict[str, list[str]] = {
        host: [] for host in schedule.host_task_queues.keys()
    }
    host_remaining = {host: tasks for host, tasks in schedule.host_task_queues.items()}

    while True:
        for host in schedule.host_task_queues.keys():
            while len(host_ongoing[host]) > 0 and executor.is_done(
                host_ongoing[host][0]
            ):
                completed = host_ongoing[host].pop(0)
                logger.debug(f"finished {completed} on {host}")
                for v in task_param_sources[id2task[completed]].values():
                    output_dependants[v].remove(id2task[completed])
                    if not output_dependants[v]:
                        logger.debug(f"{v} not needed, purging")
                        executor.purge(v[0], v[1])
            while len(host_ongoing[host]) < 2 and len(host_remaining[host]) > 0:
                nextTaskName = host_remaining[host].pop(0)
                nextTask = executable_tasks[nextTaskName]
                logger.debug(f"submitting {nextTask} on worker {host}")
                nextTaskId = executor.run_at(nextTask, host)
                host_ongoing[host].append(nextTaskId)
                id2task[nextTaskId] = nextTaskName
        ongoing = set(fut for futs in host_ongoing.values() for fut in futs)
        if len(ongoing) > 0:
            logger.debug(f"awaiting on {len(ongoing)} futures")
            executor.wait_some(ongoing)
        else:
            logger.debug("nothing runnin, breaking")
            break


class CascadeController:
    def submit(
        self,
        job: JobInstance,
        schedule: Schedule,
        executor: Executor,
        purging_policy: PurgingPolicy,
    ) -> None:
        return _submit(job, schedule, executor, purging_policy)
