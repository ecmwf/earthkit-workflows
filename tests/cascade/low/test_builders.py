# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from cascade.low.builders import JobBuilder, TaskBuilder


def test_validation():
    def test_func(x: int, y: str) -> int:
        return x + len(y)

    task_bad = TaskBuilder.from_callable(test_func).with_values(x="no", y=0)
    job = JobBuilder().with_node("task", task_bad).build()
    expected = [
        "invalid static input for task: x needs int, got <class 'str'>",
        "invalid static input for task: y needs str, got <class 'int'>",
    ]
    assert sorted(job.e) == sorted(expected)

    task_good = TaskBuilder.from_callable(test_func).with_values(x=1, y="yes")
    _ = JobBuilder().with_node("task", task_good).build().get_or_raise()

    sink_bad = TaskBuilder.from_callable(test_func).with_values(x=1)
    job = (
        JobBuilder()
        .with_node("source", task_good)
        .with_node("sink", sink_bad)
        .with_edge("source", "sink", "y")
        .build()
    )
    expected = [
        "edge connects two incompatible nodes: source=source.0 sink_task='sink' sink_input_kw='y' sink_input_ps=None",  # noqa: E501
    ]
    assert sorted(job.e) == sorted(expected)

    sink_good = TaskBuilder.from_callable(test_func).with_values(y="yes")
    _ = (
        JobBuilder()
        .with_node("source", task_good)
        .with_node("sink", sink_good)
        .with_edge("source", "sink", "x")
        .build()
        .get_or_raise()
    )
