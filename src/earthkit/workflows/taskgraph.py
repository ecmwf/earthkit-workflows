# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Callable, Iterator, cast

from typing_extensions import Self

from .graph import Graph, Node, Output
from .utility import predecessors


class Resources:
    """Record resources used by a task.

    Params
    ------
    duration: int, for duration of task in seconds
    memory: int for memory usage in MiB
    cpu_cycles: int, for cpu cycles used
    """

    def __init__(self, duration: float = 0, memory: float = 0, cpu_cycles: int = 0):
        self.duration = duration
        self.memory = memory
        self.cpu_cycles = cpu_cycles


class Task(Node):
    def __init__(
        self,
        name: str,
        outputs: list[str] | None = None,
        payload: Any = None,
        resources: Resources | None = None,
        **kwargs: Self | Output,
    ):
        super().__init__(name, outputs, payload, **kwargs)
        if resources is None:
            resources = Resources()
        self.resources = resources
        self.state = None

    @property
    def duration(self) -> float:
        return self.resources.duration

    @duration.setter
    def duration(self, value: int):
        self.resources.duration = value

    @property
    def memory(self):
        return self.resources.memory

    @memory.setter
    def memory(self, value: int):
        self.resources.memory = value

    @property
    def cpu_cycles(self):
        return self.resources.cpu_cycles

    @cpu_cycles.setter
    def cpu_cycles(self, value: int):
        self.resources.cpu_cycles = value

    def copy(self) -> "Task":
        newnode = Task(
            self.name, self.outputs.copy(), self.payload, self.resources, **self.inputs
        )
        return newnode


class Communication(Node):
    """Communication task, representing data transfer between tasks.

    Params
    ------
    name: str, name of task
    source: Node, source of transfer task
    size: float, size of transfer in MiB
    """

    def __init__(self, name: str, source: Node | Output, size: float):
        super().__init__(name, payload=None, input=source)
        self.size = size
        self.state = None


class TaskGraph(Graph):
    def __init__(self, sinks: list[Node]):
        super().__init__(sinks)
        self._accumulated_duration = {}
        for task in self.nodes(forwards=True):
            self._accumulated_duration[task] = self.accumulated_duration(task)

    def edges(self) -> Iterator[tuple[Node, Node]]:
        """Iterator over all node pairs connected by an edge in the graph.

        Returns
        -------
        Iterator[Node, Node]
        """
        for node in self.nodes():
            for input in node.inputs.values():
                yield input.parent, node

    def accumulated_duration(self, task: Node) -> float:
        """Calculate the accumulated duration of a task, using the duration of all its
        predecessors.

        Params
        ------
        task: Task

        Returns
        -------
        float, accumulated duration of task in seconds
        """
        if task in self._accumulated_duration:
            return self._accumulated_duration[task]

        duration = cast(
            Task, task
        ).duration  # nodes seem to be runtime patched to tasks
        for child in predecessors(self, task):
            if child in self._accumulated_duration:
                duration += self._accumulated_duration[child]
            else:
                duration += self.accumulated_duration(child)
        return duration


class ExecutionGraph(TaskGraph):
    def _make_communication_task(
        self, source: Node, target: Node, state: Callable | None = None
    ):
        t = Communication(
            f"{source.name}-{target.name}",
            source,
            cast(Task, source).memory,  # nodes seem to be runtime patched to tasks
        )
        t.state = state() if state is not None else None

        for iname, input in target.inputs.items():
            if input.parent.name == source.name:
                target.inputs[iname] = t.get_output()
                break
        return t
