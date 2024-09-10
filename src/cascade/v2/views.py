"""
Utility functions and transformers for the core graph objects
"""

from collections import defaultdict

from cascade.v2.core import Task2TaskEdge


def param_source(edges: list[Task2TaskEdge]) -> dict[str, dict[str, tuple[str, str]]]:
    """Returns map[sink_task][sink_input] = (source_task, source_output)"""
    rv: dict[str, dict[str, tuple[str, str]]] = defaultdict(lambda: defaultdict(lambda: {}))  # type: ignore
    for e in edges:
        rv[e.sink_task][e.sink_input] = (e.source_task, e.source_output)
    return rv
