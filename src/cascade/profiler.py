import functools
import os
import pathlib
import re
import warnings

import filelock
from memray import FileDestination, FileReader, Tracker
from meters import metered

from .executors.executor import Executor
from .fluent import Node
from .graph import Graph, Output, Transformer
from .schedulers.schedule import Schedule
from .taskgraph import Resources, Task, TaskGraph


def _memray_wrap_task(node: Node, path: pathlib.Path, native_traces: bool) -> Node:
    assert isinstance(node.payload, tuple)
    assert len(node.payload) == 3
    func = node.payload[0]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        destination = FileDestination(path, overwrite=True)
        with Tracker(destination=destination, native_traces=native_traces):
            result = func(*args, **kwargs)
        return result

    wrapped = node.copy()
    wrapped.payload = (wrapper,) + node.payload[1:]
    return wrapped


class _AddMemrayProfiler(Transformer):
    def __init__(self, base_path: pathlib.Path, native_traces: bool = False):
        self.base_path = base_path
        self.native_traces = native_traces

    def node(self, node: Node, **inputs: Output) -> Node:
        path = self.base_path / (node.name + ".bin")
        wrapped = _memray_wrap_task(node, path, self.native_traces)
        wrapped.inputs = inputs
        return wrapped

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        return graph.__class__(sinks)


class _ReadMemrayProfiles(Transformer):
    def __init__(
        self, base_path: pathlib.Path, memory: bool = True, duration: bool = True
    ):
        self.base_path = base_path
        self.memory = memory
        self.duration = duration

    def node(self, node: Node, **inputs: Output) -> Task:
        task = Task(
            node.name, node.outputs.copy(), node.payload, resources=None, **inputs
        )
        path = self.base_path / (node.name + ".bin")
        try:
            reader = FileReader(path)
        except OSError as e:
            warnings.warn(f"Could not read {path!r}: {e!s}", RuntimeWarning)
            return task
        with reader:
            if self.duration:
                task.duration = (
                    reader.metadata.end_time - reader.metadata.start_time
                ).total_seconds()
            if self.memory:
                task.memory = reader.metadata.peak_memory / (1024**2)  # Convert to MiB
        return task

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        return graph.__class__(sinks)


def _meters_wrap_task(node: Node, path: str) -> Node:
    assert isinstance(node.payload, tuple)
    assert len(node.payload) == 3
    func = node.payload[0]
    name = node.name

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        meter, result = metered(name, return_meter=True)(func)(*args, **kwargs)
        with filelock.FileLock(f"{path}.lock"):
            with open(path, "a") as f:
                f.write(f"{meter}\n")
        return result

    wrapped = node.copy()
    wrapped.payload = (wrapper,) + node.payload[1:]
    return wrapped


class _AddMetersProfiler(Transformer):
    def __init__(self, logfile: str):
        self.logfile = logfile

    def node(self, node: Node, **inputs: Output) -> Node:
        wrapped = _meters_wrap_task(node, self.logfile)
        wrapped.inputs = inputs
        return wrapped

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        return graph.__class__(sinks)


def _get_from_logline(regex: str, line: str) -> float:
    m = re.search(regex, line)
    if not m:
        raise ValueError
    return float(m.group(1))


def parse_metered_logfile(
    logfile: str, memory: bool = True, duration: bool = True
) -> dict[str, Resources]:
    resources = {}
    with open(logfile, "r") as file:
        for line in file:
            log_bytes = _get_from_logline("memory: (.+?) bytes", line)
            log_time = _get_from_logline("wall time: (.+?) s", line)
            resources[":".join(line.split(":")[0:2])] = Resources(
                log_time, log_bytes / (1024**2)
            )
    return resources


class _ReadMetersProfiles(Transformer):
    def __init__(self, logfile: str, memory: bool = True, duration: bool = True):
        self.resources = parse_metered_logfile(logfile, memory, duration)
        self.memory = memory
        self.duration = duration

    def node(self, node: Node, **inputs: Output) -> Task:
        task = Task(
            node.name, node.outputs.copy(), node.payload, resources=None, **inputs
        )
        resources = self.resources[node.name]
        if self.duration:
            task.duration = resources.duration
        if self.memory:
            task.memory = resources.memory
        return task

    def graph(self, graph: Graph, sinks: list[Node]) -> Graph:
        return graph.__class__(sinks)


def memray_profile(
    graph: Graph | Schedule,
    base_path: os.PathLike | str,
    executor: Executor,
    native_traces: bool = False,
    memory: bool = True,
    duration: bool = True,
) -> tuple[object, Graph]:
    base_path = pathlib.Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    task_graph = TaskGraph(graph.sinks) if isinstance(graph, Schedule) else graph
    wrapped_graph = _AddMemrayProfiler(base_path, native_traces).transform(task_graph)
    execution_graph = (
        Schedule(wrapped_graph, graph.context_graph, graph.task_allocation)
        if isinstance(graph, Schedule)
        else wrapped_graph
    )
    result = executor.execute(execution_graph)
    annotated_graph = _ReadMemrayProfiles(base_path, memory, duration).transform(
        task_graph
    )
    return result, annotated_graph


def meters_profile(
    graph: Graph | Schedule,
    logfile: str,
    executor: Executor,
    memory: bool = True,
    duration: bool = True,
) -> tuple[object, Graph]:
    if os.path.exists(logfile):
        os.remove(logfile)
    task_graph = TaskGraph(graph.sinks) if isinstance(graph, Schedule) else graph
    wrapped_graph = _AddMetersProfiler(logfile).transform(task_graph)
    execution_graph = (
        Schedule(wrapped_graph, graph.context_graph, graph.task_allocation)
        if isinstance(graph, Schedule)
        else wrapped_graph
    )
    result = executor.execute(execution_graph)
    annotated_graph = _ReadMetersProfiles(logfile, memory, duration).transform(
        task_graph
    )
    return result, annotated_graph


def profile(
    graph: Graph | Schedule,
    base_path: str,
    executor: Executor,
    memray_native_traces: bool = False,
    memory: str = "memray",
    duration: str = "meters",
) -> tuple[object, Graph]:
    if memory == duration == "memray":
        return memray_profile(graph, base_path, executor, memray_native_traces)
    if memory == duration == "meters":
        return meters_profile(
            graph,
            f"{base_path}/meters_log.txt",
            executor,
        )
    _, memray_graph = memray_profile(
        graph,
        base_path,
        executor,
        memray_native_traces,
        memory=(memory == "memray"),
        duration=(duration == "memray"),
    )
    return meters_profile(
        memray_graph,
        f"{base_path}/meters_log.txt",
        executor,
        memory=(memory == "meters"),
        duration=(duration == "meters"),
    )
