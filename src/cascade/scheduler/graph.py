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


def nearest_common_descendant(
    paths: Task2TaskDistance, nodes: list[TaskId], L: int
) -> Task2TaskDistance:
    ncd: Task2TaskDistance = {}
    try:
        import coptrs

        logger.debug("using coptrs library, watch out for the blazing speed")
        m = {}
        d1 = {}
        d2 = {}
        i = 0
        # TODO we convert from double dict to dict of tuples -- extend coptrs to support the other as well to get rid fo this
        for a in paths.keys():
            for b in paths[a].keys():
                if a not in d1:
                    d1[a] = i
                    d2[i] = a
                    i += 1
                if b not in d1:
                    d1[b] = i
                    d2[i] = b
                    i += 1
                m[(d1[a], d1[b])] = paths[a][b]
        ncdT: dict[tuple[int, int], int] = coptrs.nearest_common_descendant(m, L)
        for (ai, bi), e in ncdT.items():
            if d2[ai] not in ncd:
                ncd[d2[ai]] = {}
            ncd[d2[ai]][d2[bi]] = e
    except ImportError:
        logger.warning("coptrs not found, falling back to python")
        for a in nodes:
            ncd[a] = {}
            for b in nodes:
                if b == a:
                    ncd[a][b] = 0
                    continue
                ncd[a][b] = L
                for c in nodes:
                    ncd[a][b] = min(ncd[a][b], max(paths[a][c], paths[b][c]))
    return ncd


def decompose(
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


def enrich(
    plain_component: PlainComponent,
    edge_i: dict[TaskId, set[TaskId]],
    edge_o: dict[TaskId, set[TaskId]],
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

    ncd = nearest_common_descendant(paths, nodes, L)

    return ComponentCore(
        nodes=nodes,
        sources=sources,
        distance_matrix=ncd,
        value=value,
        depth=L,
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

    with ThreadPoolExecutor(max_workers=4) as tp:
        # TODO if coptrs is not used, then this doesnt make sense
        f = lambda plain_component: timer(enrich, Microtrace.presched_enrich)(
            plain_component, edge_i_proj, edge_o_proj
        )
        plain_components = (
            plain_component
            for plain_component in timer(decompose, Microtrace.presched_decompose)(
                list(job_instance.tasks.keys()),
                edge_i_proj,
                edge_o_proj,
            )
        )
        components = list(tp.map(f, plain_components))

    components.sort(key=lambda c: c.weight(), reverse=True)

    return Preschedule(components=components, edge_o=edge_o, edge_i=edge_i)
