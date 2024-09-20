from contextlib import nullcontext as does_not_raise

import pytest
from schedule_utils import example_graph

from cascade.contextgraph import ContextGraph, Processor
from cascade.schedulers.depthfirst import DepthFirstScheduler
from cascade.schedulers.schedule import Schedule
from cascade.schedulers.simulate import (
    CommunicatorState,
    ContextState,
    ExecutionState,
    Simulator,
    TaskState,
)
from cascade.taskgraph import Communication, Resources, Task
from cascade.transformers import to_task_graph


@pytest.mark.parametrize(
    "task, state, processor, end_time",
    [
        [
            Task(name="task", payload="task", resources=Resources(10, 10)),
            TaskState,
            "worker_1",
            10,
        ],
        [
            Communication(name="comm", source=Task("input"), size=10),
            CommunicatorState,
            "worker_1-worker_2",
            101,
        ],
    ],
)
def test_context_state(context_graph, task, state, processor, end_time):
    context_state = ContextState(context_graph)
    task.state = state()
    processor = context_state.processor(processor) if isinstance(task, Task) else context_state.communicator(processor)
    context_state.assign_task_to_processor(task, processor, 0, print)
    assert task.state.end_time == end_time
    assert processor.state.current_task == task
    if isinstance(processor, Processor):
        assert task in processor.state.memory_usage.current_tasks()
        assert processor.state.memory_usage.memory > 0
    assert task.state.end_time in context_state.sim.timesteps

    assert processor not in context_state.idle_processors()


def test_execution_state():
    task_graph = to_task_graph(example_graph(3), {})
    with pytest.raises(ValueError):
        ExecutionState(task_graph, with_communication=True)

    dummy_context = ContextGraph()
    dummy_context.add_node("p1", type="CPU", speed=10, memory=50)
    dummy_schedule = DepthFirstScheduler().schedule(task_graph, dummy_context)

    dummy_context.add_node("p2", type="CPU", speed=10, memory=50)
    dummy_context.add_edge("p1", "p2", bandwidth=0.1, latency=1)
    last_task = dummy_schedule.task_allocation["p1"].pop(-1)
    dummy_schedule.task_allocation["p2"] = [last_task]
    schedule = Schedule(task_graph, dummy_context, dummy_schedule.task_allocation)
    execution_state = ExecutionState(schedule, with_communication=True)
    assert execution_state.total_tasks == len(list(task_graph.nodes())) + 3
    assert len(execution_state.communication_tasks) == 3


@pytest.mark.parametrize(
    "schedule, with_communication, expectation, num_tasks",
    [
        [False, True, pytest.raises(ValueError), 0],
        [False, False, does_not_raise(), 11],
        [True, True, does_not_raise(), 14],
        [True, False, does_not_raise(), 11],
    ],
)
def test_simulator(context_graph, schedule, with_communication, expectation, num_tasks):
    task_graph = to_task_graph(example_graph(5), {})
    kwargs = {"context_graph": context_graph, "with_communication": with_communication}
    if schedule:
        task_graph = DepthFirstScheduler().schedule(task_graph, context_graph)
        kwargs.pop("context_graph")
    with expectation:
        execution_state, _ = Simulator().execute(task_graph, **kwargs)
        assert execution_state.total_tasks == num_tasks
