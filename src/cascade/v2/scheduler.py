"""
TODO replace with adapter to cascade.schedulers
"""

from collections import defaultdict
from typing import Optional

from cascade.v2.core import Environment, JobInstance, Schedule
from cascade.v2.func import Either
from cascade.v2.views import param_source


def schedule(
    job_instance: JobInstance, environment: Environment
) -> Either[Schedule, str]:
    # simplest impl: assign all to first host to produce *a* viable schedule. To be dropped soon
    if not environment.hosts:
        return Either.error("no hosts given")
    top_host = max(environment.hosts.items(), key=lambda e: e[1].memory_mb)[0]

    # TODO include optional TaskResourceRequirements, check they fit

    schedule = defaultdict(list)
    remaining = set(job_instance.tasks.keys())
    task_prereqs: dict[str, set[str]] = {
        k: set(e[0] for e in v.values())
        for k, v in param_source(job_instance.edges).items()
    }

    while remaining:
        chosen: Optional[str] = None
        for e in remaining:
            if not task_prereqs.get(e, set()).intersection(remaining):
                chosen = e
                break
        if chosen:
            schedule[top_host].append(chosen)
            remaining.remove(chosen)
        else:
            return Either.error("job instance contains a cycle")
    return Either.ok(Schedule(host_task_queues=schedule))
