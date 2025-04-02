# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Interface for tracing important events that can be used for extracting performance information

Currently, the export is handled just by logging, assuming to be parsed later. We log at debug
level since this is assumed to be high level tracing
"""

import logging
from enum import Enum
from functools import wraps
from time import perf_counter_ns, time_ns

d: dict[str, str] = {}

marker_logid = __name__ + ".marker"
marker = logging.getLogger(marker_logid)
tracer_logid = __name__ + ".tracer"
tracer = logging.getLogger(tracer_logid)
dataBegin = "#"


class TaskLifecycle(str, Enum):
    planned = "task_planned"  # controller planned the task to run at a worker
    enqueued = (
        "task_enqueued"  # worker received the task and put into its internal queue
    )
    started = "task_started"  # a process executing this task is ready and imported
    loaded = "task_loaded"  # all task's inputs are in process memory and deserialized
    computed = "task_computed"  # the task callable itself has finished
    published = (
        "task_published"  # the results have been serialized and put to shared memory
    )
    completed = "task_completed"  # the controller marked this task as completed


class TransmitLifecycle(str, Enum):
    planned = "transmit_planned"  # controller planned the transmit to run from source worker to target worker
    started = "transmit_started"  # source worker started executing the transmit
    loaded = "transmit_loaded"  # source worker has the data in memory
    received = "transmit_received"  # target worker accepted the connection
    unloaded = "transmit_unloaded"  # target worker has the data in memory
    completed = "transmit_completed"  # the controller marked this transmit as completed


class ControllerPhases(str, Enum):
    # ordered exactly as controller cycles through
    assign = "ctrl_assign"  # assignment of tasks to workers and submitting to executor, reports how many events were awaited prior
    plan = "ctrl_plan"  # planning, ie, update of schedule; reports how many actions were sent in `assign`
    flush = "ctrl_flush"  # calculate dataset purges and fetches, submit to executor
    wait = "ctrl_wait"  # await on executor results
    shutdown = "ctrl_shutdown"  # final phase so that we can calculate the duration of last wait


class Microtrace(str, Enum):
    presched_decompose = "presched_decompose"
    presched_enrich = "presched_enrich"
    ctrl_init = "ctrl_init"
    ctrl_plan = "ctrl_plan"
    ctrl_act = "ctrl_act"
    ctrl_wait = "ctrl_wait"
    ctrl_notify = "ctrl_notify"
    ctrl_assign = "ctrl_assign"
    ctrl_migrate = "ctrl_migrate"
    wrk_ser = "wrk_ser"
    wrk_deser = "wrk_deser"
    wrk_load = "wrk_load"
    wrk_compute = "wrk_compute"
    wrk_publish = "wrk_publish"
    wrk_task = "wrk_task"
    exc_procjoin = "exc_procjoin"
    total_incluster = "total_incluster"
    total_e2e = "total_e2e"


Label = int | str | Enum
Labels = dict[str, Label]


def l2s(l: Label) -> str:
    if isinstance(l, Enum):
        if isinstance(l, str):
            return l.value
        else:
            return str(l.value)
    else:
        return str(l)


def _labels(labels: Labels) -> str:
    # TODO a bit crude to call this at every mark -- precache some scoping
    return ";".join(f"{k}={l2s(v)}" for k, v in labels.items())


def label(key: str, value: str) -> None:
    """Makes all subsequent marks contain this KV. Carries over to later-forked subprocesses, but
    not to forkspawned
    """
    global d  # noqa: F824
    d[key] = value


def mark(labels: Labels) -> None:
    at = time_ns()
    global d  # noqa: F824
    event = _labels({**d, **labels})
    marker.debug(f"{dataBegin}{event};{at=}")


def trace(kind: Microtrace, value: int):
    tracer.debug(f"{dataBegin}{kind.value}={value}")


def timer(f, kind: Microtrace):
    """Don't use for distributed tracing as this relies on unsync time"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = perf_counter_ns()
        rv = f(*args, **kwargs)
        end = perf_counter_ns()
        trace(kind, end - start)
        return rv

    return wrapper
