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
    ExecutableSubgraph,
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
    required_outputs: set[tuple[str, str]],
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
        published_outputs={e for e in task.definition.output_schema if (name, e) in required_outputs},
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
        (task for subgraph in schedule.host_task_queues.values() for task in subgraph),
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
    id2tasks: dict[str, list[str]] = {}
    task2outputs: dict[str, list[tuple[str, str]]] = {
        task: [
            (task, e)
            for e in instance.definition.output_schema
        ]
        for task, instance in job.tasks.items()
    }
    task_dependants = dependants(job.edges)
    # TODO replace this and other redundants in favour of using EnvironmentState
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
                for taskName in id2tasks[completed]:
                    environment_state = environment_state.finishTaskAt(
                        host, taskName
                    )
                    # TODO this is not correct -- compute only those datasets that were actually published
                    environment_state = environment_state.computeDatasetsAt(host, task2outputs[taskName])
                    for v in set(task_param_sources[taskName].values()):
                        if not is_dataset_needed(
                            schedule, task_dependants, v, purging_policy, environment_state
                        ):
                            logger.debug(f"{v} not needed, purging")
                            for other_host in environment_state.hosts_of_ds(v):
                                executor.purge(v[0], v[1], {other_host})
                                environment_state = environment_state.purgeDatasetAt(
                                    other_host, (v[0], v[1])
                                )
            if len(host_ongoing[host]) < 1 and len(host_remaining.get(host, [])) > 0:
                nextSubgraphTaskNames = host_remaining[host].pop(0)
                nextSubgraphTasks = []
                # TODO this is wrong on two fronts:
                # - genuine outputs would not get published
                # - tasks allocated to other hosts but not running yet would not get inputs
                remainingTasks = schedule.unallocated.union({task for subgraph in host_remaining[host] for task in subgraph}) - set(nextSubgraphTaskNames)
                logger.debug(f"preparing next subgraph, with {remainingTasks=}")
                for nextTaskName in nextSubgraphTaskNames:
                    required_outputs = {
                        output
                        for output in task2outputs[nextTaskName]
                        if task_dependants[output].intersection(remainingTasks)
                        or output in purging_policy.preserve
                    }
                    nextTask = _task2executable(nextTaskName, job.tasks[nextTaskName], job, task_param_sources[nextTaskName], required_outputs)
                    reqs = {(e.sourceTask, e.sourceOutput) for e in nextTask.wirings}
                    if missing := (reqs - environment_state.ds_of_host(host, True)):
                        # TODO this should be dictated by scheduler
                        logger.debug(
                            f"task {nextTaskName} on worker {host} is {missing =}! Scattering."
                        )
                        for e in missing:
                            executor.scatter(e[0], e[1], {host})
                        environment_state = environment_state.computeDatasetsAt(host, missing)
                    logger.debug(f"preparing {nextTaskName} on worker {host} as {nextTask}")
                    nextSubgraphTasks.append(nextTask)
                    environment_state = environment_state.runTaskAt(host, nextTaskName, task2outputs[nextTaskName])
                subgraph_wrapper = ExecutableSubgraph(tasks=nextSubgraphTasks)
                logger.debug(f"submitting {':'.join(task.name for task in nextSubgraphTasks)} on worker {host}")
                nextSubgraphId = executor.run_at(subgraph_wrapper, host)
                host_ongoing[host].append(nextSubgraphId)
                id2tasks[nextSubgraphId] = nextSubgraphTaskNames
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
        dynamic_scheduler: Scheduler | None = None,
    ) -> EnvironmentState:
        if purging_policy is None:
            purging_policy = PurgingPolicy.default(job)
            logger.debug(f"defaulting {purging_policy=}")
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
