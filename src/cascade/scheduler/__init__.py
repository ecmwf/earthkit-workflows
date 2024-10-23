from cascade.low.core import Environment, JobExecutionRecord, JobInstance, Schedule
from cascade.low.func import Either
from cascade.scheduler.api import EnvironmentState, Scheduler
from cascade.scheduler.simple import BFSScheduler


def schedule(
    job_instance: JobInstance,
    environment: Environment,
    execution_record: JobExecutionRecord | None = None,
    scheduler: Scheduler | None = None,
) -> Either[Schedule, str]:
    if scheduler is None:
        scheduler = BFSScheduler()
    return scheduler.schedule(
        job_instance, environment, execution_record, EnvironmentState()
    )
