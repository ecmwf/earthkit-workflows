# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Graph decompositions and distance functions.
Used to obtain a Preschedule object from a Job Instance via the `precompute` function.
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from typing import Iterator

from cascade.low.core import DatasetId, JobInstance, TaskId
from cascade.low.tracing import Microtrace, timer
from cascade.low.views import dependants, param_source
from cascade.scheduler.core import ComponentCore, Preschedule, Task2TaskDistance

logger = logging.getLogger(__name__)

PlainComponent = tuple[list[TaskId], list[TaskId]]  # nodes, sources


def _nearest_common_descendant(
    paths: Task2TaskDistance,
    nodes: list[TaskId],
    L: int,
    parents: dict[TaskId, set[TaskId]],
    children: dict[TaskId, set[TaskId]],
) -> Task2TaskDistance:
    # well crawl through the graph starting from sinks
    remaining_children = {v: len(children[v]) for v in nodes}
    queue = [v for v in nodes if remaining_children[v] == 0]

    # for each pair of vertices V & U, we store here their so-far-nearest common descendant D + max(dist(V, D), dist(U, D))
    # we need to keep track of D while we build this to be able to recalculate, but we'll drop it in the end
    result: dict[TaskId, dict[TaskId, tuple[TaskId, int]]] = {}
    while queue:
        v = queue.pop(0)
        result[v] = {}
        # for each u, do we have a common ancestor with it?
        for u in nodes:
            # if we are their ancestor then we are a common ancestor, though not necessarily the nearest one
            if v in paths[u]:
                result[v][u] = (v, paths[u][v])
            # some of our children may have a common ancestor with u
            for c in children[v]:
                if u in result[c]:
                    d = result[c][u][0]
                    dist = max(paths[v][d], paths[u][d])
                    if u not in result[v] or result[v][u][1] > dist:
                        result[v][u] = (d, dist)
        # identify whether any of our parents children were completely processed -- if yes,
        # we can continue the crawl with them
        for p in parents[v]:
            remaining_children[p] -= 1
            if remaining_children[p] == 0:
                queue.append(p)

    # just drop the D witness, and fill default L if no common ancestor whatsoever
    ncd: Task2TaskDistance = {}
    for v in nodes:
        ncd[v] = {}
        for u in nodes:
            if u in result[v]:
                ncd[v][u] = result[v][u][1]
            else:
                ncd[v][u] = L
    return ncd


def _decompose(
    nodes: list[TaskId],
    edge_i: dict[TaskId, set[TaskId]],
    edge_o: dict[TaskId, set[TaskId]],
) -> Iterator[PlainComponent]:
    sources: set[TaskId] = {node for node in nodes if not edge_i[node]}

    sources_l: list[TaskId] = [s for s in sources]
    visited: set[TaskId] = set()

    while sources_l:
        head = sources_l.pop()
        if head in visited:
            continue
        queue: list[TaskId] = [head]
        visited.add(head)
        component: list[TaskId] = list()

        while queue:
            head = queue.pop()
            component.append(head)
            for vert in chain(edge_i[head], edge_o[head]):
                if vert in visited:
                    continue
                else:
                    visited.add(vert)
                    queue.append(vert)
        yield (
            component,
            [e for e in component if e in sources],
        )


def _enrich(
    plain_component: PlainComponent,
    edge_i: dict[TaskId, set[TaskId]],
    edge_o: dict[TaskId, set[TaskId]],
    needs_gpu: set[TaskId],
    gangs: set[TaskId],
) -> ComponentCore:
    nodes, sources = plain_component
    logger.debug(
        f"enrich component start; {len(nodes)} nodes, of that {len(sources)} sources"
    )

    sinks = [v for v in nodes if not edge_o[v]]
    remaining = {v: len(edge_o[v]) for v in nodes if edge_o[v]}
    layers: list[list[TaskId]] = [sinks]
    value: dict[TaskId, int] = {}
    paths: Task2TaskDistance = {}

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
        paths[v] = defaultdict(lambda: L)
        paths[v][v] = 0

    for layer in layers[1:]:
        for v in layer:
            value[v] = 0
            paths[v] = defaultdict(lambda: L)
            paths[v][v] = 0
            for c in edge_o[v]:
                paths[v][c] = 1
                for desc, dist in paths[c].items():
                    paths[v][desc] = min(paths[v][desc], dist + 1)
                value[v] = max(value[v], value[c] - 1)

    # calculate ncd
    ncd = _nearest_common_descendant(paths, nodes, L, edge_i, edge_o)

    # fusing opportunities
    # TODO we just arbitrarily crawl down from sinks, until everything is
    # decomposed into paths. A smarter approach would utilize profiling
    # information such as dataset size, trying to fuse the large datasets
    # first so that they end up on the longest paths
    fusing_opportunities = {}
    gpu_fused_distance = {}
    fused = set()
    while layers:
        layer = layers.pop(0)
        while layer:
            gpu_distance = None
            head = layer.pop(0)
            if head in fused or head in gangs:
                continue
            chain = []
            fused.add(head)
            found = True
            while found:
                if head in needs_gpu:
                    gpu_distance = 0
                elif gpu_distance is not None:
                    gpu_distance += 1
                gpu_fused_distance[head] = gpu_distance
                found = False
                for edge in edge_i[head]:
                    if edge not in fused and edge not in gangs:
                        chain.insert(0, head)
                        head = edge
                        fused.add(head)
                        found = True
                        break
            if len(chain) > 0:
                chain.insert(0, head)
                fusing_opportunities[head] = chain

    return ComponentCore(
        nodes=nodes,
        sources=sources,
        distance_matrix=ncd,
        value=value,
        depth=L,
        fusing_opportunities=fusing_opportunities,
        gpu_fused_distance=gpu_fused_distance,
    )


def precompute(job_instance: JobInstance) -> Preschedule:
    edge_o = dependants(job_instance.edges)
    edge_i: dict[TaskId, set[DatasetId]] = defaultdict(set)
    for task, inputs in param_source(job_instance.edges).items():
        edge_i[task] = {e for e in inputs.values()}
    edge_o_proj: dict[TaskId, set[TaskId]] = defaultdict(set)
    for dataset, outs in edge_o.items():
        edge_o_proj[dataset.task] = edge_o_proj[dataset.task].union(outs)

    edge_i_proj: dict[TaskId, set[TaskId]] = defaultdict(set)
    for vert, inps in edge_i.items():
        edge_i_proj[vert] = {dataset.task for dataset in inps}

    needs_gpu = {
        task_id
        for task_id, task in job_instance.tasks.items()
        if task.definition.needs_gpu
    }
    gangs = {
        task_id
        for constraint in job_instance.constraints
        for task_id in constraint.gang
    }

    with ThreadPoolExecutor(max_workers=4) as tp:
        # TODO if coptrs is not used, then this doesnt make sense
        f = lambda plain_component: timer(_enrich, Microtrace.presched_enrich)(
            plain_component, edge_i_proj, edge_o_proj, needs_gpu, gangs
        )
        plain_components = (
            plain_component
            for plain_component in timer(_decompose, Microtrace.presched_decompose)(
                list(job_instance.tasks.keys()),
                edge_i_proj,
                edge_o_proj,
            )
        )
        components = list(tp.map(f, plain_components))

    components.sort(key=lambda c: c.weight(), reverse=True)

    return Preschedule(components=components, edge_o=edge_o, edge_i=edge_i)
