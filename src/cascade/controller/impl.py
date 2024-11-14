import logging
from typing import Any
from cascade.low.core import JobInstance, DatasetId, Environment, WorkerId, TaskId
from cascade.low.views import param_source, dependants
from cascade.scheduler.core import Schedule
from cascade.controller.executor import Executor
from cascade.controller.core import State, Action, Event, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit, TaskStatus
from cascade.controller.notify import notify
from cascade.controller.act import act
from cascade.controller.plan import plan
from cascade.controller.views import colocated_workers
from time import perf_counter_ns
from cascade.controller.tracing import mark
from cascade.low.func import assert_never, simple_timer

logger = logging.getLogger(__name__)

def tracingReport(actions: list[Action], tAct: int, tWait: int|None, remainingTasks: dict[WorkerId, TaskStatus], events: list[Event]) -> None:
    # NOTE this way won't yield easily total idle time -- a worker is idle if reported here as idle for `tWait` _and_ for the next report's `tAct`
    # NOTE worker is considered busy *iff* it has any task assigned -- but the task may only be enqueued
    d: dict[str, Any] = {"host": "controller", "action": "controllerReport", "actDuration": tAct, "waitDuration": tWait, "actionsTransmit": 0, "actionsSubmit": 0, "actionsPurge": 0, "eventsTransmited": 0, "eventsComputed": 0, "eventsStarted": 0}
    for action in actions:
        if isinstance(action, ActionDatasetPurge):
            d["actionsPurge"] += 1
        elif isinstance(action, ActionDatasetTransmit):
            d["actionsTransmit"] += 1
        elif isinstance(action, ActionSubmit):
            d["actionsSubmit"] += 1
        else:
            assert_never(action)
    for event in events:
        for ts in event.ts_trans:
            if ts[1] == TaskStatus.running:
                d["eventsStarted"] += 1
            elif ts[1] == TaskStatus.succeeded:
                d["eventsComputed"] += 1
        if not event.ts_trans:
            for ds in event.ds_trans:
                d["eventsTransmited"] += 1
    d["busyWorkers"] = len(set(remainingTasks.keys()))
    d["progressingTasks"] = len(remainingTasks.values())
    mark(d)

def run(job: JobInstance, executor: Executor, schedule: Schedule) -> State:
    env = executor.get_environment()
    logger.debug(f"starting with {env=}")
    paramSource = param_source(job.edges)
    taskInputs = {
        task_id: set(taskParamSource.values())
        for task_id, taskParamSource in paramSource.items()
    }
    purging_tracker = dependants(job.edges)
    state = State(purging_tracker, colocated_workers(env))

    try:
        while schedule.layers or state.remaining:
            # plan
            actions = plan(schedule, state, env, job, taskInputs)
            # act
            if actions:
                state, tAct = simple_timer(act)(executor, state, actions)
            remaining_tasks = {
                worker: status
                for task in state.remaining
                for worker, status in state.ts2worker[task].items()
            }
            tWait = None
            # wait
            if remaining_tasks:
                logger.debug(f"about to await executor because of {remaining_tasks=}")
                events, tWait = simple_timer(executor.wait_some)()
                notify(state, events, taskInputs)
                logger.debug(f"received {len(events)} events")
            tracingReport(actions, tAct, tWait, remaining_tasks, events)
    except Exception:
        logger.error("crash in controller, shuting down")
        raise
    finally:
        logger.debug(f"shutting down executor")
        executor.shutdown()
    return state
