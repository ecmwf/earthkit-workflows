from .scheduler import Schedule, DepthFirstScheduler
from .graphs import ContextGraph, TaskGraph
from .executor import ExecutionReport, BasicExecutor


class Cascade:

    def create_schedule(taskgraph: TaskGraph, contextgraph: ContextGraph) -> Schedule:
        return DepthFirstScheduler(taskgraph, contextgraph).create_schedule()
    
    def execute(schedule: Schedule) -> ExecutionReport:
        return BasicExecutor(schedule).execute()
    
    def simulate(schedule: Schedule, with_communication: bool = True) -> ExecutionReport:
        return BasicExecutor(schedule, with_communication).simulate()
    
    # TODO: maybe add an API that allows constructing the graphs here.



            



