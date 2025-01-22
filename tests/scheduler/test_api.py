"""
Tests calculation of preschedule, state initialize & first assign and plan
"""

from cascade.low.core import DatasetId, WorkerId
from cascade.scheduler.api import initialize, assign, plan
from cascade.scheduler.core import Assignment, TaskStatus
from cascade.scheduler.graph import precompute
from .util import get_env, get_job0, get_job1

    # def assign(state: State) -> Iterator[Assignment]:
    # def plan(state: State, assignments: list[Assignment]) -> State:

def test_job0():
    job0, _ = get_job0()
    preschedule = precompute(job0)

    h1w1 = get_env(1, 1)
    h1w1_w = WorkerId("h0", "w0")
    state = initialize(h1w1, preschedule, set())
    assignment = list(assign(state))
    assert assignment == [
        Assignment(worker=h1w1_w, tasks=['source'], prep=[], outputs={DatasetId(task='source', output='0')})
    ]

    state = plan(state, assignment)
    assert state.worker2ts == {h1w1_w: {'source': TaskStatus.enqueued}}

def test_job1():
    job1, _ = get_job1()
    preschedule = precompute(job1)

    h1w1 = get_env(1, 1)
    h1w1_w = WorkerId("h0", "w0")
    state = initialize(h1w1, preschedule, set())
    assignment = list(assign(state))
    assert assignment == [
        Assignment(worker=h1w1_w, tasks=['source'], prep=[], outputs={DatasetId(task='source', output='0')})
    ]

    state = plan(state, assignment)
    assert state.worker2ts == {h1w1_w: {'source': TaskStatus.enqueued}}

# TODO add some multi-source or multi-component job
