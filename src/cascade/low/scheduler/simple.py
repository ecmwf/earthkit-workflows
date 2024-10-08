"""
Simple Scheduling algorithms:
1. bfs schedule
 - schedules all (remaining) tasks
 - BFS/round robin fashion among workers
 - ignores affinities, memory restrictions, etc

2. dfs one worker schedule
 - schedules all (remaining) tasks
 - DFS to single worker
 - ignores affinities, memory restrictions, etc

3. sink bfs redundant schedule
 - schedules all (remaining) tasks
 - round robin assigns each sink (leaf) to workers
 - each worker bfs-es on the path towards sinks
 - if a node lies on multiple paths, each worker computes its own
 - ignores affinities, memory restrictions, etc
"""

import logging
from collections import defaultdict
from itertools import cycle
from typing import cast

from cascade.low.core import Environment, JobExecutionRecord, JobInstance, Schedule
from cascade.low.func import Either, maybe_head
from cascade.low.scheduler.api import ClasslessScheduler, EnvironmentState
from cascade.low.views import dependants, param_source

logger = logging.getLogger(__name__)


def bfs_schedule(
    job_instance: JobInstance,
    environment: Environment,
    execution_record: JobExecutionRecord | None,
    environment_state: EnvironmentState,
) -> Either[Schedule, str]:
    schedule: dict[str, list[str]] = defaultdict(list)
    remaining = set(job_instance.tasks.keys()) - environment_state.finished_tasks()

    task_prereqs: dict[str, set[str]] = {
        k: set(e[0] for e in v.values())
        for k, v in param_source(job_instance.edges).items()
    }

    while remaining:
        computable: list[str] = []
        for e in remaining:
            if not task_prereqs.get(e, set()).intersection(remaining):
                computable.append(e)
        if not computable:
            return Either.error("job instance contains a cycle")
        for t, h in zip(computable, cycle(environment.hosts)):
            schedule[h].append(t)
            remaining.remove(t)
    return Either.ok(Schedule(host_task_queues=schedule))


BFSScheduler = lambda: ClasslessScheduler(bfs_schedule)


def dfs_one_worker_schedule(
    job_instance: JobInstance,
    environment: Environment,
    execution_record: JobExecutionRecord | None,
    environment_state: EnvironmentState,
) -> Either[Schedule, str]:
    schedule: dict[str, list[str]] = defaultdict(list)
    host = maybe_head(environment.hosts)
    if host is None:
        return Either.error("this scheduler compatible with exactly one worker only")

    task_v_in: dict[str, set[str]] = {
        k: set(e[0] for e in v.values())
        for k, v in param_source(job_instance.edges).items()
    }
    schedule = defaultdict(list)

    computed = environment_state.finished_tasks()
    remaining = set(job_instance.tasks.keys()) - computed
    touched = set()

    def visit(node: str) -> None:
        if node in computed:
            return
        if node in touched:
            raise ValueError("cycle")
        touched.add(node)
        for v_in in task_v_in.get(node, set()):
            visit(v_in)
        computed.add(node)
        remaining.remove(node)
        schedule[host].append(node)

    while True:
        candidate = maybe_head(remaining)
        if candidate is None:
            break
        visit(candidate)

    return Either.ok(Schedule(host_task_queues=schedule))


DFSOneWorkerScheduler = lambda: ClasslessScheduler(dfs_one_worker_schedule)


def sink_bfs_redundant_schedule(
    job_instance: JobInstance,
    environment: Environment,
    execution_record: JobExecutionRecord | None,
    environment_state: EnvironmentState,
) -> Either[Schedule, str]:
    task_v_in: dict[str, set[str]] = {
        k: set(e[0] for e in v.values())
        for k, v in param_source(job_instance.edges).items()
    }
    task_v_out: dict[str, set[str]] = {
        k[0]: v for k, v in dependants(job_instance.edges).items()
    }
    schedule: dict[str, list] = defaultdict(list)
    finished_tasks = environment_state.finished_tasks()

    sinks = {k for k in job_instance.tasks.keys() if not task_v_out.get(k, set())}
    for host, sink in zip(cycle(environment.hosts), sinks):

        reachable = {sink}
        reachable_queue = list(task_v_in[sink])
        while reachable_queue:
            h = reachable_queue.pop(0)
            if h in reachable:
                continue
            for v in task_v_in[h]:
                if v in reachable:
                    continue
                else:
                    reachable.add(v)
                    reachable_queue.append(v)

        irrelevant = set(job_instance.tasks) - reachable

        queue = task_v_in[sink]
        computed = {sink}
        rev_schedule = [sink]
        while queue:
            next_batch = []
            for e in queue:
                if (task_v_out[e] - irrelevant) <= computed:
                    next_batch.append(e)
                    computed.add(e)
            if not next_batch:
                raise ValueError
            rev_schedule.extend(next_batch)
            for e in next_batch:
                queue.remove(e)
                queue.update(task_v_in.get(e, set()))
        schedule[host].extend(e for e in rev_schedule[::-1] if e not in finished_tasks)
    for host in schedule:
        # we make the schedules not contain the same task twice -- it could have happened
        # if multiple sinks got assigned to the same host, and they shared source(s)
        uniq = set()
        schedule[host] = [
            e for e in schedule[host] if e not in uniq and not cast(bool, uniq.add(e))
        ]
    return Either.ok(Schedule(host_task_queues=schedule))


SinkBFSRedundantScheduler = lambda: ClasslessScheduler(sink_bfs_redundant_schedule)
