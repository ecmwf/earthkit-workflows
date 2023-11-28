from typing import Any

from .graph import Graph
from .graph import Node


class Task(Node):
    def __init__(
        self,
        name: str,
        outputs: list[str] | None = None,
        payload: Any = None,
        **kwargs: "Node | Node.Output",
    ):
        super().__init__(name, outputs, payload, **kwargs)
        self.cost = 0
        self.in_memory = 0
        self.out_memory = 0
        self.state = None

    @property
    def memory(self):
        return max(self.in_memory, self.out_memory)

    def copy(self) -> "Task":
        newnode = Task(self.name, self.outputs.copy(), self.payload, **self.inputs)
        newnode.cost = self.cost
        newnode.in_memory = self.in_memory
        newnode.out_memory = self.out_memory
        return newnode


class Communication(Node):
    def __init__(self, name, source, size):
        super().__init__(name, payload=None, input=source)
        self.size = size
        self.state = None


class TaskGraph(Graph):
    def predecessors(self, task) -> list[Task]:
        return [
            x if isinstance(x, Task) else x[0]
            for x in self.get_predecessors(task).values()
        ]

    def successors(self, task) -> list[Task]:
        return [x[0] for x in sum(self.get_successors(task).values(), [])]

    def edges(self):
        for node in self.nodes():
            for input in node.inputs.values():
                yield input.parent, node


class ExecutionGraph(TaskGraph):
    def _make_communication_task(self, source, target):
        t = Communication(f"{source.name}-{target.name}", source, source.out_memory)

        for iname, input in target.inputs.items():
            if input.name == source:
                target.inputs[iname] = t.get_output()
                break
        return t
