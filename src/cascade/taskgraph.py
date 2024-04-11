from typing import Any

from .graph import Graph
from .graph import Node, Sink
from .utility import predecessors


class Resources:
    def __init__(self, cost: float = 0, memory: float = 0):
        """
        Params
        ------
            cost: int, for duration of task in ms
            memory: int for memory usage in MB
        """
        self.cost = cost
        self.memory = memory


class Task(Node):
    def __init__(
        self,
        name: str,
        outputs: list[str] | None = None,
        payload: Any = None,
        **kwargs: "Node | Node.Output",
    ):
        super().__init__(name, outputs, payload, **kwargs)
        self.resources = Resources()
        self.state = None

    @property
    def cost(self):
        return self.resources.cost

    @cost.setter
    def cost(self, value: int):
        self.resources.cost = value

    @property
    def memory(self):
        return self.resources.memory

    @memory.setter
    def memory(self, value: int):
        self.resources.memory = value

    def copy(self) -> "Task":
        newnode = Task(self.name, self.outputs.copy(), self.payload, **self.inputs)
        newnode.cost = self.cost
        newnode.memory = self.memory
        return newnode


class Communication(Node):
    def __init__(self, name, source, size):
        super().__init__(name, payload=None, input=source)
        self.size = size
        self.state = None


class TaskGraph(Graph):
    def __init__(self, sinks: list[Sink]):
        super().__init__(sinks)
        self._accumulated_cost = {}
        for task in self.nodes(forwards=True):
            self._accumulated_cost[task] = self.accumulated_cost(task)

    def edges(self):
        for node in self.nodes():
            for input in node.inputs.values():
                yield input.parent, node

    def accumulated_cost(self, task: Task) -> float:
        if task in self._accumulated_cost:
            return self._accumulated_cost[task]

        cost = task.cost
        for child in predecessors(self, task):
            if child in self._accumulated_cost:
                cost += self._accumulated_cost[child]
            else:
                cost += self.accumulated_cost(child)
        return cost


class ExecutionGraph(TaskGraph):
    def _make_communication_task(self, source, target):
        t = Communication(f"{source.name}-{target.name}", source, source.memory)

        for iname, input in target.inputs.items():
            if input.name == source:
                target.inputs[iname] = t.get_output()
                break
        return t
