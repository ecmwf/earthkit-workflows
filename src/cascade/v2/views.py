"""
Utility functions and transformers for the core graph objects
"""

from collections import defaultdict

from cascade.v2.core import Task2TaskEdge


def param_source(
    edges: list[Task2TaskEdge],
) -> dict[str, dict[int | str, tuple[str, str]]]:
    """Returns map[sink_task][sink_input] = (source_task, source_output)"""
    rv: dict[str, dict[int | str, tuple[str, str]]] = defaultdict(lambda: defaultdict(lambda: {}))  # type: ignore
    for e in edges:
        sink_input: int | str
        if e.sink_input_kw is not None:
            if e.sink_input_ps is not None:
                raise TypeError
            else:
                sink_input = e.sink_input_kw
        else:
            if e.sink_input_ps is None:
                raise TypeError
            else:
                sink_input = e.sink_input_ps
        rv[e.sink_task][sink_input] = (e.source_task, e.source_output)
    return rv
