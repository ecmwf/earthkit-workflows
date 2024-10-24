import logging
from cascade.low.core import Environment, JobExecutionRecord, JobInstance, TaskId, DatasetId
from cascade.low.views import param_source
from cascade.scheduler.core import Schedule
from cascade.low.func import Either, maybe_head

logger = logging.getLogger(__name__)

def naive_bfs_layers(job: JobInstance, record: JobExecutionRecord, completed: set[TaskId]) -> Either[Schedule, str]:
    """Ignores record, decomposes graphs into bfs layers. Order within layers arbitrary"""

    layers: list[list[TaskId]] = []
    task_prereqs: dict[TaskId, set[TaskId]] = {
        task_id: {source.task for source in taskParamSource.values()}
        for task_id, taskParamSource in param_source(job.edges).items()
    }
    remaining: set[TaskId] = set(job.tasks.keys()) - completed

    # TODO proper topo decompo algo
    while remaining:
        logger.debug(f"{remaining=}")
        this_layer: list[str] = []
        for e in remaining:
            if not task_prereqs.get(e, set()).intersection(remaining):
                this_layer.append(e)
        if not this_layer:
            return Either.error("job instance contains a cycle")
        layers.append(this_layer)
        for e in this_layer:
            remaining.remove(e)
    return Either.ok(Schedule(layers=layers))

def naive_dfs_layers(job: JobInstance, record: JobExecutionRecord, completed: set[TaskId]) -> Either[Schedule, str]:
    """Ignores record, decomposes graphs into single-element dfs layers. Order of layers arbitrary"""
    layers: list[list[TaskId]] = []
    task_prereqs: dict[TaskId, set[TaskId]] = {
        task_id: {source.task for source in taskParamSource.values()}
        for task_id, taskParamSource in param_source(job.edges).items()
    }
    remaining = set(job.tasks.keys()) - completed

    computed: set[TaskId] = set()
    touched: set[TaskId] = set()
    def visit(node: TaskId) -> None:
        if node in computed:
            return
        if node in touched:
            raise ValueError("cycle")
        touched.add(node)
        for v_in in task_prereqs.get(node, set()):
            visit(v_in)
        computed.add(node)
        remaining.remove(node)
        layers.append([node])

    while True:
        candidate = maybe_head(remaining)
        if candidate is None:
            break
        visit(candidate)

    return Either.ok(Schedule(layers=layers))
