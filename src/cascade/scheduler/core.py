from cascade.low.core import TaskId, DatasetId
from dataclasses import dataclass

Task2TaskDistance = dict[TaskId, dict[TaskId, int]]

TaskValue = dict[TaskId, int]

@dataclass
class Component:
    nodes: list[TaskId]
    sources: list[TaskId]
    weight: int
    distance_matrix: Task2TaskDistance # nearest common descendant
    value: TaskValue # closer to a sink -> higher value

@dataclass
class Preschedule:
    components: list[Component] # sorted desc by weight
    edge_o: dict[DatasetId, set[TaskId]]
    edge_i: dict[TaskId, set[DatasetId]]


# Worker2TaskDistance

# WorkerState // dataset x state

#ScheduleComponent
#    component
#    computable: [node]
#    remaining: [node] or {node} // do we even need it?
#    worker2task_distance
#    
#
#ScheduleGraph
