from collections import defaultdict
from dataclasses import dataclass, field

from cascade.low.core import TaskId, DatasetId

Task2TaskDistance = dict[TaskId, dict[TaskId, int]]

TaskValue = dict[TaskId, int]

@dataclass
class ComponentCore:
    nodes: list[TaskId]
    sources: list[TaskId]
    weight: int # of *remaining* tasks
    distance_matrix: Task2TaskDistance # nearest common descendant
    value: TaskValue # closer to a sink -> higher value
    depth: int # maximum value

@dataclass
class Preschedule:
    components: list[Component] # sorted desc by weight
    edge_o: dict[DatasetId, set[TaskId]]
    edge_i: dict[TaskId, set[DatasetId]]


Worker2TaskDistance = dict[WorkerId, dict[TaskId, int]]

ComponentId = int

@dataclass
class ComponentSchedule:
    core: ComponentCore
    worker2task_distance: Worker2TaskDistance = field(default_factory=defaultdict(dict))
    computable: dict[TaskId, int] # task & optimum distance attained by some worker

class DatasetStatus(int, Enum):
    missing = -1 # virtual default status, never stored
    preparing = 0 # set by controller
    available = 1 # set by executor
    purged = 2 # temporal command status used as local comms between controller.act and controller.state

class TaskStatus(int, Enum):
    enqueued = 0 # set by controller
    running = 1 # set by executor
    succeeded = 2 # set by executor
    failed = 3 # set by executor

@dataclass
class State:
    """Captures what is where -- datasets, running tasks, ... Used for decision making and progress tracking"""
    # TODO separate into two structures: lookups and trackers? We have controller leaking in here

    # lookups
    edge_o: dict[DatasetId, set[TaskId]]
    edge_i: dict[TaskId, set[DatasetId]]
    worker2ds: dict[WorkerId, dict[DatasetId, DatasetStatus]] = field(default_factory=lambda: defaultdict(dict))
    ds2worker: dict[DatasetId, dict[WorkerId, DatasetStatus]] = field(default_factory=lambda: defaultdict(dict))
    ts2worker: dict[TaskId, dict[WorkerId, TaskStatus]] = field(default_factory=lambda: defaultdict(dict))
    worker2ts: dict[WorkerId, dict[TaskId, TaskStatus]] = field(default_factory=lambda: defaultdict(dict))
    host2ds: dict[HostId, dict[DatasetId, DatasetStatus]] = field(default_factory=lambda: defaultdict(dict))
    ds2host: dict[DatasetId, dict[HostId, DatasetStatus]] = field(default_factory=lambda: defaultdict(dict))

    # schedule -- updated by scheduler.api.{assign, plan}
    components: list[ComponentSchedule]
    host2component: dict[HostId, ComponentId]
    computable: int
    worker2taskOverhead: Worker2TaskDistance = field(default_factory=defaultdict(dict))

    # trackers
    idle_workers: set[WorkerId] = field(default_factory=set) # add by controller.notify, remove by scheduler.api.assign
    ongoing: set[TaskId] = field(default_factory=set) # add by controller.act, remove by controller.notify
    purging_tracker: dict[DatasetId, set[TaskId]] # add by controller.notify, remove by controller.act
    purging_queue: list[DatasetId] = field(default_factory=list) # TODO is even necessary?
    outputs: dict[DatasetId, Any] # key add by scheduler.api.init, value add by controller.notify
    fetching_queue: dict[DatasetId, WorkerId] = field(default_factory=dict) # add by controller.notify, remove by controller.act

    # TODO redundant via host2ds, eliminate
    worker_colocations: dict[WorkerId, set[WorkerId]] 

def has_computable(state: State) -> bool:
    return state.computable > 0

def has_awaitable(state: State) -> bool:
    # TODO replace the None in outputs with check on fetch queue (but change that from binary to ternary first)
    return state.running or (None in state.outputs.values()):

@dataclass
class Assignment:
    worker: WorkerId
    tasks: list[TaskId]
    prep: list[tuple[DatasetId, WorkerId]]
