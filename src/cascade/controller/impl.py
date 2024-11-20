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
from cascade.controller.tracing import mark, label, ControllerPhases, Microtrace, timer
from cascade.low.func import assert_never

logger = logging.getLogger(__name__)

def summarise_actions(actions: list[Action]) -> dict:
    d = {"actionsPurge": 0, "actionsTransmit": 0, "actionsSubmit": 0}
    for action in actions:
        if isinstance(action, ActionDatasetPurge):
            d["actionsPurge"] += 1
        elif isinstance(action, ActionDatasetTransmit):
            d["actionsTransmit"] += 1
        elif isinstance(action, ActionSubmit):
            d["actionsSubmit"] += 1
        else:
            assert_never(action)
    return d

def summarise_events(events: list[Event]) -> dict:
    d = {"eventsStarted": 0, "eventsComputed": 0, "eventsTransmited": 0}
    for event in events:
        for ts in event.ts_trans:
            if ts[1] == TaskStatus.running:
                d["eventsStarted"] += 1
            elif ts[1] == TaskStatus.succeeded:
                d["eventsComputed"] += 1
        if not event.ts_trans:
            for ds in event.ds_trans:
                d["eventsTransmited"] += 1
    return d

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
    label("host", "controller")
    events: list[Event] = []

    try:
        while schedule.computable or state.remaining:
            mark({"action": ControllerPhases.plan, **summarise_events(events)})
            actions = timer(plan, Microtrace.ctrl_plan)(schedule, state, env, job, taskInputs)

            mark({"action": ControllerPhases.act, **summarise_actions(actions)})
            if actions:
                state = timer(act, Microtrace.ctrl_act)(executor, state, actions)
            tWait = None

            mark({"action": ControllerPhases.wait})
            if state.remaining:
                logger.debug(f"about to await executor because of {state.remaining=}")
                events = timer(executor.wait_some, Microtrace.ctrl_wait)()
                timer(notify, Microtrace.ctrl_notify)(state, events, taskInputs)
                logger.debug(f"received {len(events)} events")
    except Exception:
        logger.error("crash in controller, shuting down")
        raise
    finally:
        mark({"action": ControllerPhases.shutdown})
        logger.debug(f"shutting down executor")
        executor.shutdown()
    return state
