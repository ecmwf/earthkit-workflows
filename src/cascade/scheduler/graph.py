"""
Graph decompositions and distance functions.
Used to obtain a Preschedule object from a Job Instance via the `precompute` function.
"""

from collections import defaultdict
import logging
from itertools import chain
from cascade.low.core import TaskId, DatasetId, JobInstance
from cascade.low.views import dependants, param_source
from typing import Iterator
from cascade.scheduler.core import Task2TaskDistance, TaskValue, ComponentCore, Preschedule

logger = logging.getLogger(__name__)

PlainComponent = tuple[list[TaskId], list[TaskId]] # nodes, sources

def decompose(nodes: list[TaskId], edge_i: dict[TaskId, set[TaskId]], edge_o: dict[TaskId, set[TaskId]]) -> Iterator[PlainComponent]:
    sources: set[TaskId] = {
        node
        for node in nodes
        if not edge_i[node]
    }

    sources_l: list[TaskId] = [s for s in sources]
    visited: set[TaskId] = set()

    while sources_l:
        head = sources_l.pop()
        if head in visited:
            continue
        queue: list[TaskId] = [head]
        component: list[TaskId] = list()

        while queue:
            head = queue.pop()
            component.append(head)
            for vert in chain(edge_i[head], edge_o[head]):
                if vert in visited:
                    continue
                else:
                    visited.add(vert)
                    component.append(vert)
                    queue.append(vert)
        yield (
            component,
            [e for e in component if e in sources],
        )

def enrich(plain_component: PlainComponent, edge_i: dict[TaskId, set[TaskId]], edge_o: dict[TaskId, set[TaskId]]) -> ComponentCore:
    nodes, sources = plain_component

    sinks = [v for v in nodes if not edge_o[v]]
    remaining = {v: len(edge_o[v]) for v in nodes if edge_o[v]}
    layers: list[list[TaskId]] = [sinks]
    value: dict[TaskId, int] = {}
    paths: Task2TaskDistance = {}
    ncd: Task2TaskDistance = {}

    # decompose into topological layers
    while remaining:
        next_layer = []
        for v in layers[-1]:
            for a in edge_i[v]:
                remaining[a] -= 1
                if remaining[a] == 0:
                    next_layer.append(a)
                    remaining.pop(a)
        layers.append(next_layer)

    L = len(layers)

    # calculate value, ie, inv distance to sink
    for v in layers[0]:
        value[v] = L
        paths[v] = defaultdict(lambda : L)
        paths[v][v] = 0
        ncd[v] = defaultdict(lambda : L)

    for layer in layers[1:]:
        for v in layer:
            value[v] = 0
            paths[v] = defaultdict(lambda : L)
            paths[v][v] = 0
            for c in edge_o[v]:
                paths[v][c] = 1
                for desc, dist in paths[c].items():
                    paths[v][desc] = min(paths[v][desc], dist+1)
                value[v] = max(value[v], value[c] - 1)

    # calculate nearest common descendant
    # NOTE sorta floyd warshall, ie, n3. Rewrite into n2?
    for a in nodes:
        ncd[a] = {}
        for b in nodes:
            if b == a:
                ncd[a][b] = 0
                continue
            ncd[a][b] = L
            for c in nodes:
                ncd[a][b] = min(ncd[a][b], max(paths[a][c], paths[b][c]))
                
    return ComponentCore(
        nodes=nodes,
        sources=sources,
        distance_matrix=ncd,
        value=value,
        depth=L,
    )

def precompute(job_instance: JobInstance) -> Preschedule:
    edge_o: dict[DatasetId, set[TaskId]] = dependants(job_instance.edges)
    edge_o_proj: dict[TaskId, set[TaskId]] = defaultdict(set)
    for dataset, outs in edge_o.items():
        edge_o_proj[dataset.task] = edge_o_proj[dataset.task].union(outs)

    edge_i: dict[TaskId, set[DatasetId]] = defaultdict(set)
    for task, inputs in param_source(job_instance.edges).items():
        edge_i[task] = {e for e in inputs.values()}
    edge_i_proj: dict[TaskId, set[TaskId]] = defaultdict(set)
    for vert, inps in edge_i.items():
        edge_i_proj[vert] = {dataset.task for dataset in inps}

    task_o = {
        task: job_instance.outputs_of(task)
        for task in job_instance.tasks.keys()
    }

    components = [
        enrich(plain_component, edge_i_proj, edge_o_proj)
        for plain_component in decompose(
            list(job_instance.tasks.keys()),
            edge_i_proj,
            edge_o_proj,
        )
    ]
    components.sort(key=lambda c: c.weight(), reverse=True)

    return Preschedule(components=components, edge_o=edge_o, edge_i=edge_i, task_o=task_o)
