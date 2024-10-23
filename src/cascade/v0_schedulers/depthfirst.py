import functools
from typing import cast

from cascade.v0_contextgraph import ContextGraph, Processor
from cascade.graph import Graph
from cascade.taskgraph import TaskGraph
from cascade.transformers import to_task_graph

from .schedule import Schedule
from .simulate import ContextState, ExecutionState, Simulator


class DepthFirstScheduler(Simulator):
    def __init__(self):
        super().__init__()
        self.task_allocation = {}

    def reset_state(self):
        super().reset_state()
        self.task_allocation = {}

    def assign_eligible_tasks(
        self, time: float, execution_state: ExecutionState, context_state: ContextState
    ) -> None:
        # NOTE a few typing glitches here wrt processor-communicator, should be cleared up
        context_state.update(self.completed_tasks)
        for processor in context_state.idle_processors():
            if not hasattr(processor, "state"):
                raise TypeError
            mem_usage = processor.state.memory_usage.memory
            pop_index = None
            # Take from back so newly added dependents get picked off first
            for index in range(
                len(self.eligible[Simulator.DEFAULT_PROCESSOR]) - 1, -1, -1
            ):
                processor_ = cast(
                    Processor, self.eligible[Simulator.DEFAULT_PROCESSOR][index]
                )
                if (mem_usage + processor_.memory) < cast(Processor, processor).memory:
                    pop_index = index
                    break

            if pop_index is not None:
                next_task = self.eligible[Simulator.DEFAULT_PROCESSOR].pop(pop_index)
                self.task_allocation.setdefault(processor.name, []).append(
                    next_task.name
                )
                context_state.assign_task_to_processor(
                    next_task,
                    cast(Processor, processor),
                    time,
                    functools.partial(
                        self.on_task_complete,
                        execution_state,
                        context_state,
                        next_task,
                        cast(Processor, processor),
                    ),
                )

    def initialise_eligible_tasks(self, graph: TaskGraph | Schedule) -> None:
        graph = cast(TaskGraph, graph)  # not sure if we shouldnt raise instead
        # Sort sinks by total duration
        sinks = graph.sinks[:]
        sinks.sort(key=lambda x: graph.accumulated_duration(x))
        self.eligible[Simulator.DEFAULT_PROCESSOR] = []
        for sink in sinks:
            tmp_graph = Graph([sink])
            for src in tmp_graph.sources():
                if src not in self.eligible[Simulator.DEFAULT_PROCESSOR]:
                    self.eligible[Simulator.DEFAULT_PROCESSOR].append(src)
        self.eligible[Simulator.DEFAULT_PROCESSOR].reverse()

    def schedule(
        self,
        task_graph: Graph | TaskGraph,
        context_graph: ContextGraph,
    ) -> Schedule:
        """Schedule tasks in task graph to workers in context graph.

        Params
        ------
        task_graph: Graph or TaskGraph, if Graph then will perform execution of graph
        using thread pool to determine resources in the transformation to a TaskGraph
        context_graph: ContextGraph, containers nodes to which tasks should be assigned

        Returns
        -------
        Schedule
        """

        if not isinstance(task_graph, TaskGraph):
            task_graph = to_task_graph(task_graph)
        super().execute(
            task_graph,
            context_graph=context_graph,
            with_communication=False,
        )
        return Schedule(task_graph, context_graph, self.task_allocation)
