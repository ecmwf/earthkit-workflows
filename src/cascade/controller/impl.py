"""
Implements the canonical cascade controller
"""

# TODO major improvements
# - smarter scatter -- do beforehand, dictate by scheduler
# - smarter purges -- per-worker, dictate by scheduler
# - support for multiple outputs / generators per task
# - control precisely which futures to launch already -- currently we just throw in first in the queue per host
# - handle failed states, restarts, etc

import logging
from functools import reduce
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


def is_dataset_needed(
    schedule: Schedule,
    dependants: dict[tuple[str, str], set[str]],
    dataset: tuple[str, str],
    purging_policy: PurgingPolicy,
    environment_state: EnvironmentState,
) -> bool:
    if not purging_policy.eager or dataset in purging_policy.preserve:
        return True
    # TODO we recompute this from scratch because of the changing schedule. Wasteful, fix
    remaining_tasks = reduce(
        lambda acc, e: acc.union(e),
        schedule.host_task_queues.values(),
        schedule.unallocated,
    ).union({e for _, e in environment_state.runningTaskAtHost})
    # TODO we should make this more fine-grained -- now we either keep all copies, or discard all copies
    return dependants[dataset].intersection(remaining_tasks) != set()


def _submit(
    job: JobInstance,
    schedule: Schedule,
    executor: Executor,
    execution_record: JobExecutionRecord,
    purging_policy: PurgingPolicy,
    environment_state: EnvironmentState,
    dynamic_scheduler: Scheduler | None,
) -> EnvironmentState:
    task_param_sources = param_source(job.edges)
    executable_tasks = {
        task: _task2executable(task, instance, job, task_param_sources[task])
        for task, instance in job.tasks.items()
    }
    id2task: dict[str, str] = {}
    task2outputs: dict[str, list[tuple[str, str]]] = {
        task: [
            (task, e)
            for e in instance.definition.output_schema
        ]
        for task, instance in job.tasks.items()
    }
    task_dependants = dependants(job.edges)
    host_ongoing: dict[str, list[str]] = {
        host: [] for host in executor.get_environment().hosts
    }
    host_remaining = {host: queue for host, queue in schedule.host_task_queues.items()}

    while True:
        logger.debug(f"{host_ongoing=}")
        logger.debug(f"{host_remaining=}")
        for host in executor.get_environment().hosts.keys():
            logger.debug(f"checking {host=} for completions")
            while len(host_ongoing[host]) > 0 and executor.is_done(
                host_ongoing[host][0]
            ):
                completed = host_ongoing[host].pop(0)
                logger.debug(f"finished {completed} on {host}")
                environment_state = environment_state.finishTaskAt(
                    host, id2task[completed]
                )
                for output in task2outputs[id2task[completed]]:
                    environment_state = environment_state.computeDatasetAt(host, output)
                for v in set(task_param_sources[id2task[completed]].values()):
                    if not is_dataset_needed(
                        schedule, task_dependants, v, purging_policy, environment_state
                    ):
                        logger.debug(f"{v} not needed, purging")
                        for other_host in environment_state.hosts_of_ds(v):
                            executor.purge(v[0], v[1], {other_host})
                            environment_state = environment_state.purgeDatasetAt(
                                other_host, (v[0], v[1])
                            )
            while len(host_ongoing[host]) < 1 and len(host_remaining.get(host, [])) > 0:
                nextTaskName = host_remaining[host].pop(0)
                nextTask = executable_tasks[nextTaskName]
                reqs = {(e.sourceTask, e.sourceOutput) for e in nextTask.wirings}
                if missing := (reqs - environment_state.ds_of_host(host)):
                    # TODO this should be dictated by scheduler
                    logger.debug(
                        f"task {nextTaskName} on worker {host} is {missing =}! Scattering."
                    )
                    for e in missing:
                        executor.scatter(e[0], e[1], {host})
                        environment_state = environment_state.computeDatasetAt(
                            host, (e[0], e[1])
                        )
                logger.debug(f"submitting {nextTaskName} on worker {host}")
                nextTaskId = executor.run_at(nextTask, host)
                environment_state = environment_state.runTaskAt(host, nextTaskName)
                host_ongoing[host].append(nextTaskId)
                id2task[nextTaskId] = nextTaskName
        ongoing = set(fut for futs in host_ongoing.values() for fut in futs)
        if len(ongoing) > 0:
            logger.debug(f"awaiting on {len(ongoing)} futures {ongoing}")
            executor.wait_some(ongoing)
        if dynamic_scheduler is not None and (
            not host_remaining.values()
            or min(len(v) for v in host_remaining.values()) == 0
        ):
            logger.debug("idle worker, asking dynamic scheduler for more")
            schedule = dynamic_scheduler.schedule(
                job, executor.get_environment(), execution_record, environment_state
            ).get_or_raise()
            logger.debug(f"obtained new schedule {schedule}")
            host_remaining = {
                host: queue for host, queue in schedule.host_task_queues.items()
            }
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
        dynamic_scheduler: DynamicScheduler | None = None,
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
            dynamic_scheduler,
        )
