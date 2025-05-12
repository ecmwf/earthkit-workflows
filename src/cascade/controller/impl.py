# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import cascade.executor.serde as serde
from cascade.controller.act import act, flush_queues
from cascade.controller.core import State, init_state
from cascade.controller.notify import notify
from cascade.controller.report import Reporter
from cascade.executor.bridge import Bridge, Event
from cascade.low.core import JobInstance, type_dec
from cascade.low.execution_context import init_context
from cascade.low.tracing import ControllerPhases, Microtrace, label, mark, timer
from cascade.scheduler.api import assign, init_schedule, plan
from cascade.scheduler.core import Preschedule

logger = logging.getLogger(__name__)


def run(
    job: JobInstance,
    bridge: Bridge,
    preschedule: Preschedule,
    report_address: str | None = None,
) -> State:
    env = bridge.get_environment()
    context = init_context(env, job, preschedule.edge_o, preschedule.edge_i)
    outputs = set(context.job_instance.ext_outputs)
    logger.debug(f"starting with {env=} and {report_address=}")
    schedule = timer(init_schedule, Microtrace.ctrl_init)(preschedule, context)
    state = init_state(outputs, context.edge_o)

    label("host", "controller")
    events: list[Event] = []
    for serdeTypeEnc, (serdeSer, serdeDes) in context.job_instance.serdes.items():
        serde.SerdeRegistry.register(type_dec(serdeTypeEnc), serdeSer, serdeDes)
    reporter = Reporter(report_address)

    try:
        while (
            state.has_awaitable()
            or context.has_awaitable()
            or schedule.has_computable()
        ):
            mark({"action": ControllerPhases.assign})
            assignments = []
            if schedule.has_computable():
                for assignment in assign(schedule, context):
                    timer(act, Microtrace.ctrl_act)(bridge, assignment)
                    assignments.append(assignment)

            mark({"action": ControllerPhases.plan})
            plan(schedule, context, assignments)
            mark({"action": ControllerPhases.flush})
            flush_queues(bridge, state, context)

            mark({"action": ControllerPhases.wait})
            if state.has_awaitable() or context.has_awaitable():
                logger.debug(f"about to await bridge with {context.ongoing_total=}")
                events = timer(bridge.recv_events, Microtrace.ctrl_wait)()
                timer(notify, Microtrace.ctrl_notify)(
                    state, schedule, context, events, reporter
                )
                logger.debug(f"received {len(events)} events")
    except Exception as ex:
        logger.error("crash in controller, shuting down")
        reporter.send_failure(repr(ex))
        raise
    else:
        reporter.success()
    finally:
        mark({"action": ControllerPhases.shutdown})
        logger.debug("shutting down executors")
        bridge.shutdown()
    return state
