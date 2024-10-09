"""
Implements the canonical cascade controller
"""

# TODO major improvements
# - scatter variables to the workers that will need them, adapt accordingly purging policy
# - support for multiple outputs / generators per task
# - control precisely which futures to launch already -- currently we just throw in first in the queue per host
# - handle failed states, restarts, etc

import logging
from typing import cast

from cascade.controller.api import (
    ExecutableTaskInstance,
    Executor,
    PurgingPolicy,
    VariableWiring,
)
from cascade.low.core import JobExecutionRecord, JobInstance, Schedule, TaskInstance
from cascade.low.func import maybe_head
from cascade.low.scheduler.api import EnvironmentState, Scheduler
from cascade.low.scheduler.dynamic import DynamicScheduler
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
    execution_record: JobExecutionRecord,
    purging_policy: PurgingPolicy,
    environment_state: EnvironmentState,
    dynamic_scheduler: Scheduler,
) -> EnvironmentState:
    task_param_sources = param_source(job.edges)
    executable_tasks = {
        task: _task2executable(task, instance, job, task_param_sources[task])
        for task, instance in job.tasks.items()
    }
    id2task: dict[str, str] = {}
    task2output: dict[str, tuple[str, str]] = {
        task: (task, cast(str, maybe_head(instance.definition.output_schema)))
        for task, instance in job.tasks.items()
    }

    output_dependants: dict[tuple[str, tuple[str, str]], set[str]] = {}
    if purging_policy.eager:
        output_dependants = {
            (h, k): {e for e in v if e in schedule.host_task_queues[h]}
            for h in schedule.host_task_queues.keys()
            for k, v in dependants(job.edges).items()
            if v
            and k not in purging_policy.preserve
            and k[0] in schedule.host_task_queues[h]
        }
    host_ongoing: dict[str, list[str]] = {
        host: [] for host in schedule.host_task_queues.keys()
    }
    host_remaining = {host: tasks for host, tasks in schedule.host_task_queues.items()}
    logger.debug(f"{output_dependants=}")

    while True:
        logger.debug(f"{host_ongoing=}")
        for host in executor.get_environment().hosts.keys():
            while len(host_ongoing[host]) > 0 and executor.is_done(
                host_ongoing[host][0]
            ):
                completed = host_ongoing[host].pop(0)
                logger.debug(f"finished {completed} on {host}")
                environment_state = environment_state.finishTaskAt(
                    host, id2task[completed]
                )
                environment_state = environment_state.computeDatasetAt(
                    host, task2output[id2task[completed]]
                )
                # NOTE we need the `set`, because a task can use the same input twice
                for v in set(task_param_sources[id2task[completed]].values()):
                    output_dependants[(host, v)].remove(id2task[completed])
                    if not output_dependants[(host, v)]:
                        logger.debug(f"{v} not needed, purging")
                        executor.purge(v[0], v[1], {host})
                        environment_state = environment_state.purgeDatasetAt(
                            host, (v[0], v[1])
                        )
            while len(host_ongoing[host]) < 1 and len(host_remaining[host]) > 0:
                nextTaskName = host_remaining[host].pop(0)
                nextTask = executable_tasks[nextTaskName]
                logger.debug(f"submitting {nextTaskName} on worker {host}")
                nextTaskId = executor.run_at(nextTask, host)
                environment_state = environment_state.runTaskAt(host, nextTaskName)
                host_ongoing[host].append(nextTaskId)
                id2task[nextTaskId] = nextTaskName
        ongoing = set(fut for futs in host_ongoing.values() for fut in futs)
        if len(ongoing) > 0:
            logger.debug(f"awaiting on {len(ongoing)} futures {ongoing}")
            executor.wait_some(ongoing)
        if min(len(v) for v in host_remaining.values()) == 0:
            logger.debug("idle worker, asking dynamic scheduler for more")
            schedule = dynamic_scheduler.schedule(
                job, executor.get_environment(), execution_record, environment_state
            ).get_or_raise()
        if len(environment_state.finished_tasks()) == len(job.tasks):
            logger.debug("everything complete, breaking")
            break
    return environment_state


class CascadeController:
    def submit(
        self,
        job: JobInstance,
        schedule: Schedule,
        executor: Executor,
        execution_record: JobExecutionRecord | None = None,
        purging_policy: PurgingPolicy | None = None,
    ) -> EnvironmentState:
        if purging_policy is None:
            purging_policy = PurgingPolicy()
        if execution_record is None:
            execution_record = JobExecutionRecord()
        return _submit(
            job,
            schedule,
            executor,
            execution_record,
            purging_policy,
            EnvironmentState(),
            DynamicScheduler(),
        )
