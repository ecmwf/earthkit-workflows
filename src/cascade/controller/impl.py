import logging
from typing import Any
from cascade.low.core import JobInstance, DatasetId, Environment, WorkerId, TaskId
from cascade.low.views import param_source, dependants
from cascade.scheduler.core import Preschedule
from cascade.scheduler.api import initialize
from cascade.controller.executor import Executor
from cascade.controller.core import State, Action, Event, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit, TaskStatus
from cascade.controller.notify import notify
from cascade.controller.act import act
from cascade.controller.plan import plan
from time import perf_counter_ns
from cascade.low.tracing import mark, label, ControllerPhases, Microtrace, timer
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

def run(job: JobInstance, executor: Executor, preschedule: Prechedule, outputs: set[DatasetId]|None = None) -> State:
    if outputs is None:
        outputs = set()
    env = executor.get_environment()
    logger.debug(f"starting with {env=}")
    state = timer(initialize, Microtrace.ctrl_init)(env, preschedule, outputs)
    label("host", "controller")
    events: list[Event] = []

    try:
        while has_computable(state) or has_awaitable(state):
            mark({"action": ControllerPhases.assign, **summarise_events(events)})
            if has_computable(state):
                actions = []
                for action in assign(state, events):
                    timer(act, Microtrace.ctrl_act)(executor, action)
                    actions.append(action)

            mark({"action": ControllerPhases.ctrl_plan, **summarise_events(actions)})
            state = plan(state, job, actions)

            mark({"action": ControllerPhases.wait})
            if has_awaitable(state):
                logger.debug(f"about to await executor")
                events = timer(executor.wait_some, Microtrace.ctrl_wait)()
                timer(notify, Microtrace.ctrl_notify)(state, events)
                logger.debug(f"received {len(events)} events")
    except Exception:
        logger.error("crash in controller, shuting down")
        raise
    finally:
        mark({"action": ControllerPhases.shutdown})
        logger.debug(f"shutting down executor")
        executor.shutdown()
    return state
