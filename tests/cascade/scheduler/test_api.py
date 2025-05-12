# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests calculation of preschedule, state initialize & first assign and plan"""

from cascade.low.core import DatasetId, WorkerId
from cascade.low.execution_context import TaskStatus, init_context
from cascade.scheduler.api import assign, init_schedule, plan
from cascade.scheduler.core import Assignment
from cascade.scheduler.graph import precompute

from .util import get_env, get_job0, get_job1

# def assign(state: State) -> Iterator[Assignment]:
# def plan(state: State, assignments: list[Assignment]) -> State:


def test_job0():
    job0, _ = get_job0()
    preschedule = precompute(job0)

    h1w1 = get_env(1, 1)
    h1w1_w = WorkerId("h0", "w0")
    context = init_context(h1w1, job0, preschedule.edge_o, preschedule.edge_i)
    schedule = init_schedule(preschedule, context)
    assignment = list(assign(schedule, context))
    assert assignment == [
        Assignment(
            worker=h1w1_w,
            tasks=["source"],
            prep=[],
            outputs={DatasetId(task="source", output="0")},
        )
    ]

    plan(schedule, context, assignment)
    assert context.worker2ts == {h1w1_w: {"source": TaskStatus.enqueued}}


def test_job1():
    job1, _ = get_job1()
    preschedule = precompute(job1)

    h1w1 = get_env(1, 1)
    h1w1_w = WorkerId("h0", "w0")
    context = init_context(h1w1, job1, preschedule.edge_o, preschedule.edge_i)
    schedule = init_schedule(preschedule, context)
    assignment = list(assign(schedule, context))
    assert assignment == [
        Assignment(
            worker=h1w1_w,
            tasks=["source"],
            prep=[],
            outputs={DatasetId(task="source", output="0")},
        )
    ]

    plan(schedule, context, assignment)
    assert context.worker2ts == {h1w1_w: {"source": TaskStatus.enqueued}}


# TODO add some multi-source or multi-component job
