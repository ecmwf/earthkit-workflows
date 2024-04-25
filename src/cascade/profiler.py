import functools
import os
import pathlib
import warnings

from memray import FileReader, Tracker

from .executors.dask import DaskLocalExecutor
from .fluent import Node
from .graph import Graph, Transformer
from .taskgraph import Task


def _wrap_task(node: Node, path: pathlib.Path) -> Node:
    assert isinstance(node.payload, tuple)
    assert len(node.payload) == 3
    func = node.payload[0]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Tracker(path, native_traces=True):
            result = func(*args, **kwargs)
        return result

    wrapped = node.copy()
    wrapped.payload = (wrapper,) + node.payload[1:]
    return wrapped


class _AddProfiler(Transformer):
    def __init__(self, base_path: pathlib.Path):
        self.base_path = base_path

    def node(self, node: Node, **inputs: Node.Output) -> Node:
        path = self.base_path / (node.name + ".bin")
        wrapped = _wrap_task(node, path)
        wrapped.inputs = inputs
        return wrapped

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        return graph.__class__(sinks)


class _ReadProfiles(Transformer):
    def __init__(self, base_path: pathlib.Path):
        self.base_path = base_path

    def node(self, node: Node, **inputs: Node.Output) -> Task:
        task = Task(node.name, node.outputs.copy(), node.payload, **inputs)
        path = self.base_path / (node.name + ".bin")
        try:
            reader = FileReader(path)
        except OSError as e:
            warnings.warn(f"Could not read {path!r}: {e!s}", RuntimeWarning)
            return task
        with reader:
            task.cost = (
                reader.metadata.end_time - reader.metadata.start_time
            ).total_seconds()
            task.memory = reader.metadata.peak_memory
        return task

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        return graph.__class__(sinks)


def profile(
    graph: Graph, base_path: os.PathLike, *args, **kwargs
) -> tuple[object, Graph]:
    base_path = pathlib.Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    wrapped_graph = _AddProfiler(base_path).transform(graph)
    result = DaskLocalExecutor(*args, **kwargs).execute(wrapped_graph)
    annotated_graph = _ReadProfiles(base_path).transform(graph)
    return result, annotated_graph
