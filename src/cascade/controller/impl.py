import logging
from cascade.low.core import JobInstance, DatasetId
from cascade.low.views import param_source, dependants
from cascade.scheduler.core import Schedule
from cascade.controller.executor import Executor
from cascade.controller.core import State
from cascade.controller.notify import notify
from cascade.controller.act import act
from cascade.controller.plan import plan

logger = logging.getLogger(__name__)

def run(job: JobInstance, executor: Executor, schedule: Schedule) -> State:
    env = executor.get_environment()
    paramSource = param_source(job.edges)
    taskInputs = {
        task_id: set(taskParamSource.values())
        for task_id, taskParamSource in paramSource.items()
    }
    purging_tracker = dependants(job.edges)
    state = State(purging_tracker)

    while True:
        actions = plan(schedule, state, env, job, taskInputs)
        if actions:
            act(executor, state, actions)
        if state.remaining:
            logger.debug("about to await executor because of {state.remaining}")
            events = executor.wait_some()
            logger.debug(f"received {len(events)} events")
            notify(state, events, taskInputs)
        if not schedule.layers and not state.remaining:
            break

    return state
