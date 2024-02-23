from dask.utils import apply

from .graph import Sink, Node, Graph, Transformer
from .taskgraph import Resources, Task, TaskGraph, ExecutionGraph
from .contextgraph import ContextGraph


def determine_resources(graph: Graph, num_workers: int = 1) -> dict[str, Resources]:
    """
    Determines CPU and memory resources used by tasks in graph by executing the
    task graph using a thread pool.

    Params
    ------
    graph: Graph, task graph to determine resources for
    num_workers: int, number of threads in thread pool. Default is 1.

    Returns
    -------
    dict[str, Resources], dictionary containing Resource object for each node name
    """
    from pproc.common.resources import metered
    from .schedulers.depthfirst import DepthFirstScheduler
    from concurrent.futures import ThreadPoolExecutor

    task_graph = to_task_graph(graph, {})
    context = ContextGraph()
    for index in range(num_workers):
        context.add_node(f"cpu_{index}", type="CPU", speed=500, memory=500)
    schedule = DepthFirstScheduler().schedule(task_graph, context)

    resources = {}
    futures = {}
    ready = (x[0] for x in schedule.task_allocation.values())
    completed_tasks = []
    with ThreadPoolExecutor(num_workers) as executor:
        while len(completed_tasks) != len(list(graph.nodes())):
            for task_name in ready:
                task = task_graph.get_node(task_name)
                payload = task.payload
                futures[task_name] = executor.submit(
                    metered(return_meter=True)(payload[0]), *payload[1], **payload[2]
                )
            ready = set()

            new_futures = {}
            for task_name, future in futures.items():
                if future.done():
                    meter, output = future.result()
                    resources[task_name] = Resources(meter.elapsed_cpu, meter.mem)

                    task = task_graph.get_node(task_name)
                    # Pass output to arguments in payloads requiring it
                    successors = task_graph.successors(task)
                    for successor in successors:
                        assert not isinstance(
                            successor.payload, str
                        ), f"Payload can not be str. Got {successor.payload}"

                        task_ready = True
                        for iname, input in successor.inputs.items():
                            if input.parent == task:
                                successor.payload = tuple(
                                    output if (isinstance(x, str) and x == iname) else x
                                    for x in successor.payload
                                )
                            else:
                                if input.parent.name not in completed_tasks:
                                    task_ready = False
                        if task_ready:
                            ready.add(successor.name)

                    task.payload = None
                    completed_tasks.append(task_name)
                else:
                    new_futures[task_name] = future
            future = new_futures

    return resources


class _ToTaskGraph(Transformer):
    def __init__(self, resource_map: dict[str, Resources]):
        self.resource_map = resource_map

    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = Task(node.name, node.outputs.copy(), node.payload)
        newnode.inputs = inputs
        newnode.resources = self.resource_map.get(node.name, Resources())
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> TaskGraph:
        return TaskGraph(sinks)


def to_task_graph(
    graph: Graph, resource_map: dict[str, Resources] | None = {}
) -> TaskGraph:
    """
    Transform graph into task graph, with resource allocation for each task.

    Params
    ------
    graph: Graph to transform
    resource_map: dict or None, if None then resources are determined from executing graph
    using thread pool

    Returns
    -------
    TaskGraph
    """
    if resource_map is None:
        # TODO: import resource meter without eccode dependence
        # resource_map = determine_resources(graph)
        import warnings

        warnings.warn(
            "Determining resources not implemented. Revert to empty resources",
            UserWarning,
        )
        resource_map = {}
    return _ToTaskGraph(resource_map).transform(graph)


class _ToExecutionGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Task:
        newnode = node.copy()
        newnode.inputs = inputs
        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> ExecutionGraph:
        return ExecutionGraph(sinks)


def to_execution_graph(graph: Graph) -> ExecutionGraph:
    return _ToExecutionGraph().transform(graph)


class _ToDaskGraph(Transformer):
    def node(self, node: Node, **inputs: Node.Output) -> Node:
        new_payload = list(node.payload)
        new_payload[1] = list(new_payload[1])
        for input_name, input in inputs.items():
            if input_name in new_payload[1]:
                new_payload[1][new_payload[1].index(input_name)] = input.parent.name
        newnode = node.copy()
        newnode.payload = tuple([apply] + new_payload)  # Need apply for kwargs
        newnode.inputs = inputs

        return newnode

    def graph(self, graph: Graph, sinks: list[Sink]) -> dict:
        new_graph = Graph(sinks)
        ret = {}
        new_nodes = list(new_graph.nodes(forwards=True))
        for node in new_nodes:
            assert node.name not in ret, f"{node.name} already exists"
            ret[node.name] = node.payload
        assert list(ret.keys()) == [
            node.name for node in new_nodes
        ], f"Expected {len(new_nodes)} nodes, got {len(ret.keys())}"
        return ret


def to_dask_graph(graph: Graph) -> dict:
    return _ToDaskGraph().transform(graph)
