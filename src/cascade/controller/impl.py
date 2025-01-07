import logging
from typing import Any
from cascade.low.core import JobInstance, DatasetId, Environment, WorkerId, TaskId
from cascade.low.views import param_source, dependants
from cascade.scheduler.core import Preschedule, has_computable, has_awaitable, Assignment
from cascade.scheduler.api import initialize, assign, plan
from cascade.executor.bridge import Bridge
from cascade.controller.core import State, Event, TaskStatus
from cascade.controller.notify import notify
from cascade.controller.act import act, flush_queues
from time import perf_counter_ns
from cascade.low.tracing import mark, label, ControllerPhases, Microtrace, timer
from cascade.low.func import assert_never

logger = logging.getLogger(__name__)

def summarise_assignment(assignment: Assignment) -> dict:
    # NOTE this should be reworked... track inside bridge instead?
    d = {"actionsPurge": 0, "actionsTransmit": 0, "actionsSubmit": 0}
    
    for assignment in assignments:
        d["actionsSubmit"] += 1
        d["actionsTransmit"] += len(assignment.prep)
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

def run(job: JobInstance, bridge: Bridge, preschedule: Preschedule, outputs: set[DatasetId]|None = None) -> State:
    if outputs is None:
        outputs = set()
    env = bridge.get_environment()
    logger.debug(f"starting with {env=}")
    state = timer(initialize, Microtrace.ctrl_init)(env, preschedule, outputs)
    label("host", "controller")
    events: list[Event] = []

    try:
        while has_computable(state) or has_awaitable(state):
            mark({"action": ControllerPhases.assign, **summarise_events(events)})
            assignments = []
            if has_computable(state):
                for assignment in assign(state):
                    timer(act, Microtrace.ctrl_act)(bridge, state, assignment)
                    assignments.append(assignment)

            mark({"action": ControllerPhases.plan, **summarise_assignment(assignments)})
            state = plan(state, assignments)
            mark({"action": ControllerPhases.flush})
            state = flush_queues(bridge, state)

            mark({"action": ControllerPhases.wait})
            if has_awaitable(state):
                logger.debug(f"about to await bridge")
                events = timer(bridge.recv_events, Microtrace.ctrl_wait)()
                timer(notify, Microtrace.ctrl_notify)(state, events)
                logger.debug(f"received {len(events)} events")
    except Exception:
        logger.error("crash in controller, shuting down")
        raise
    finally:
        mark({"action": ControllerPhases.shutdown})
        logger.debug(f"shutting down executors")
        bridge.shutdown()
    return state
