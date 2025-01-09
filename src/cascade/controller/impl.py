import logging
from typing import Any
from cascade.low.core import JobInstance, DatasetId, Environment, WorkerId, TaskId
from cascade.low.views import param_source, dependants
from cascade.scheduler.core import Preschedule, has_computable, has_awaitable, Assignment
from cascade.scheduler.api import initialize, assign, plan
from cascade.executor.bridge import Bridge, Event
from cascade.scheduler.core import State, TaskStatus
from cascade.controller.notify import notify
from cascade.controller.act import act, flush_queues
from time import perf_counter_ns
from cascade.low.tracing import mark, label, ControllerPhases, Microtrace, timer
from cascade.low.func import assert_never

logger = logging.getLogger(__name__)

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
            mark({"action": ControllerPhases.assign})
            assignments = []
            if has_computable(state):
                for assignment in assign(state):
                    timer(act, Microtrace.ctrl_act)(bridge, state, assignment)
                    assignments.append(assignment)

            mark({"action": ControllerPhases.plan})
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
