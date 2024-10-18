"""
Simple Dynamic Scheduler
 - won't necessarily schedule everything -- only so that all workers are (reasonably) occupied
 - a worker is considered eligible for more work iff it has less tasks than cpus, and at least 512MB
   (MB is calculated from running task mem overhead estimate and stored datasets, *not* by checking worker)
 - a task is considered eligible if all its inputs have been computed already, anywhere in the cluster
 - we iterate over eligible tasks, and select an eligible worker (if available) such that
     - firstly, the estimated memory overcommit is the lowest (ideally 0 -- we dont optimize for maximising free space)
     - secondly, the total amount of datasets to be transfered is the lowest
 - no eligible worker is allocated more tasks -- once a task is found, the worker is considered ineligible
 - the schedule is returned -- note that empty schedule means that either no task has all inputs computed, or
   all workers are busy computing. In either case, the controller should simply `await` more and then call this again
"""

# TODO this is essentially stateless implementation -- however, it would benefit
# from JobInstance being decomposed once at the execution beginning in some topological
# layers, and have the affinities/weights updated dynamically. Otherwise this wont scale

# TODO a better implementation instead matches each worker with one (uncomputed) leaf, starting
# from the most-lightweight leaves, and consistenly aims to stick the path to the leaf to the worker.
# Only if no progress can be made would a worker switch over to a different leaf. Under good
# circumstances, the current implementation would converge to exactly that, but its not guaranteed.
# An intermediate step would just order eligible task in order of their salience, ie, the weight/size
# of the subtrees they unlock, the shallowest & lightest first

import time
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import cast

from cascade.low.core import (
    Environment,
    Host,
    JobExecutionRecord,
    JobInstance,
    Schedule,
)
from cascade.low.func import Either, maybe_head
from cascade.low.scheduler.api import EnvironmentState
from cascade.low.views import param_source

logger = logging.getLogger(__name__)


@dataclass
class Config:
    worker_eligible_mem_threshold: int = 512
    worker_eligible_cpu_threshold: int = 1


def _host_free_mb_cpu(
    host_datasets: set[tuple[str, str]],
    host_tasks: set[str],
    execution_record: JobExecutionRecord,
    host: Host,
) -> tuple[int, int]:
    cpu = host.cpu - len(
        host_tasks
    )  # NOTE crude -- better score would consider task runtimes or remaining cpusec
    mbs_ts = sum((execution_record.tasks[t].memory_mb for t in host_tasks), 0)
    mbs_ds = sum((execution_record.datasets_mb[d] for d in host_datasets), 0)
    logger.debug(f"host {host} has {mbs_ts=} and {mbs_ds=}")
    return cpu, host.memory_mb - mbs_ts - mbs_ds


def dynamic_schedule(
    job_instance: JobInstance,
    environment: Environment,
    execution_record: JobExecutionRecord,
    environment_state: EnvironmentState,
    config: Config,
) -> Either[Schedule, str]:
    schedule = defaultdict(list)
    remaining = set(job_instance.tasks.keys()) - environment_state.started_tasks()
    if len(remaining) == 0:
        return Either.ok(Schedule(host_task_queues=schedule))
    host_ds_ts = {
        host: environment_state.ds_and_ts_of_host(host) for host in environment.hosts
    }
    host_free_mb_cpu = {
        host: _host_free_mb_cpu(
            host_ds_ts[host][0],
            host_ds_ts[host][1],
            execution_record,
            environment.hosts[host],
        )
        for host in environment.hosts
    }

    logger.debug(f"{host_free_mb_cpu=}")
    ok_hosts = {
        host
        for host in environment.hosts
        if host_free_mb_cpu[host][0] >= config.worker_eligible_mem_threshold
        and host_free_mb_cpu[host][1] >= config.worker_eligible_cpu_threshold
    }

    if not ok_hosts:
        logger.debug("unable to find any eligible worker")
        return Either.ok(Schedule(host_task_queues=schedule, unallocated=remaining))

    task_prereqs: dict[str, set[str]] = {
        k: set((e[0], e[1]) for e in v.values())
        for k, v in param_source(job_instance.edges).items()
    }
    available_ds = environment_state.available_datasets()
    logger.debug(f"{available_ds=}")
    logger.debug(f"{remaining=}")
    runnable = {
        task for task in remaining if task_prereqs.get(task, set()) <= available_ds
    }
    logger.debug(f"{runnable=}")

    for task in runnable:
        task_outputs = execution_record.datasets_mb.get(
            (
                task,
                cast(
                    str, maybe_head(job_instance.tasks[task].definition.output_schema)
                ),
            ),
            0,
        )
        task_cost = execution_record.tasks[task].memory_mb + task_outputs
        maybe_host = None
        maybe_transf = sys.maxsize
        maybe_misfit = sys.maxsize
        for host in ok_hosts:
            misfit = task_cost - host_free_mb_cpu[host][0]
            missing_ds = task_prereqs.get(task, set()) - host_ds_ts[host][0]
            transf = sum((execution_record.datasets_mb[e] for e in missing_ds), 0)
            if misfit < 0:
                misfit = 0
            logger.debug(f"considering {host=} for {task=} -> {misfit=}, {transf=}")
            if maybe_misfit >= misfit and maybe_transf > transf:
                maybe_misfit = misfit
                maybe_transf = transf
                maybe_host = host
        if maybe_host:
            schedule[host].append(task)
            ok_hosts.remove(host)

    unallocated = remaining - {task for queue in schedule.values() for task in queue}
    logger.debug(f"finished with {schedule=}, {unallocated=}")
    return Either.ok(Schedule(host_task_queues=schedule, unallocated=unallocated))


class DynamicScheduler:
    def __init__(self, config: Config):
        self.config = config
        self.cum_time = 0

    def schedule(
        self,
        job_instance: JobInstance,
        environment: Environment,
        execution_record: JobExecutionRecord | None,
        environment_state: EnvironmentState,
    ) -> Either[Schedule, str]:
        if not environment.hosts:
            return Either.error("no hosts given")

        this_start = time.perf_counter_ns()
        res = dynamic_schedule(
            job_instance, environment, execution_record, environment_state, self.config
        )
        this_took = time.perf_counter_ns() - this_start
        logger.debug(f"this run of dyn sched took {this_took / 1e9: .3f}")
        self.cum_time += this_took
        return res
