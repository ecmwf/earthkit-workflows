import multiprocess as mp
import functools
import threading
from typing import Any

from meters import metered
from cascade.schedulers.schedule import Schedule
from cascade.graph import Graph, Node
from cascade.taskgraph import Resources


def execute_task(results, function, args, kwargs):
    extracted_args = [results[arg] if arg in results else arg for arg in args]
    return function(*extracted_args, **kwargs)


def worker(inqueue: mp.SimpleQueue, outqueue: mp.SimpleQueue):
    while True:
        next = inqueue.get()
        if next is None:
            return

        fn, args, kwargs, callback, error_callback = next
        try:
            result = (True, fn(*args, **kwargs))
        except Exception as e:
            result = (False, e)
        outqueue.put((result, callback, error_callback))


class WorkerPool:
    def __init__(self, workers: list[str]):
        self._inqueues = {name: mp.SimpleQueue() for name in workers}
        self._outqueue = mp.SimpleQueue()
        self._pool = [
            mp.Process(target=worker, args=(self._inqueues[name], self._outqueue))
            for name in workers
        ]
        for process in self._pool:
            process.start()

        self.result_handler = threading.Thread(
            target=WorkerPool._handle_results, args=(self._outqueue,)
        )
        self.result_handler.start()

    def submit(
        self,
        worker: str,
        fn: callable,
        args: list,
        kwargs: dict | None = None,
        callback=None,
        error_callback=None,
    ):
        if kwargs is None:
            kwargs = {}
        self._inqueues[worker].put((fn, args, kwargs, callback, error_callback))

    def _handle_results(outqueue: mp.SimpleQueue):
        while True:
            next = outqueue.get()
            if next is None:
                return

            result, callback, error_callback = next
            if result[0]:
                callback(result[1])
            else:
                error_callback(result[1])

    def terminate(self):
        for inqueue in self._inqueues.values():
            inqueue.put(None)
        self._outqueue.put(None)

        for p in self._pool:
            if p.exitcode is None:
                p.terminate()

        self.result_handler.join()

        for p in self._pool:
            if p.is_alive():
                p.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


def successors(graph: Graph | Schedule, task: Node) -> list[Node]:
    if isinstance(graph, Graph):
        return [x[0] for x in sum(graph.get_successors(task).values(), [])]
    return graph.successors(task)


class ProcessPoolExecutor:
    class State:
        def __init__(self):
            self.pending = []
            self.results = mp.Manager().dict()
            self.resources = {}

        def eligible(self, schedule: Schedule | Graph) -> set[Node]:
            if isinstance(schedule, Schedule):
                sources = set(
                    schedule.task_graph.get_node(x[0])
                    for x in schedule.task_allocation.values()
                )
                graph = schedule.task_graph
            else:
                sources = set(x for x in schedule.nodes() if x.is_source())
                graph = schedule

            if len(self.pending) == 0 and len(self.results) == 0:
                return sources

            # Determine next tasks ready for submission
            ready = set()
            index = 0
            while index < len(self.pending):
                task_name = self.pending[index]
                task = graph.get_node(task_name)
                if task_name in self.results:
                    next_tasks = successors(schedule, task)
                    for next in next_tasks:
                        task_ready = True
                        for input in next.inputs.values():
                            if input.parent.name not in self.results:
                                task_ready = False
                        if task_ready and next.name not in self.results:
                            ready.add(next)
                    self.pending.pop(index)
                else:
                    index += 1
            return ready

        def clear_cache(self, task_graph: Graph):
            tasks = list(self.results.keys())
            for task_name in tasks:
                if self.results[task_name] is not None:
                    task = task_graph.get_node(task_name)
                    if (
                        all(
                            [
                                x.name in self.results
                                for x in successors(task_graph, task)
                            ]
                        )
                        and task not in task_graph.sinks
                    ):
                        self.results[task_name] = None

    def _on_task_complete(state: "ProcessPoolExecutor.State", task_name: str, result):
        if task_name in state.results:
            raise ValueError(f"Task {task_name} already completed")
        meter, result = result
        state.results[task_name] = result
        # Convert memory to MB
        state.resources[task_name] = Resources(meter.elapsed_cpu, meter.mem / 10**6)

    def _on_task_error(task_name: str, result):
        error, result = result
        raise Exception(error)

    def _execute_schedule(schedule: Schedule):
        state = ProcessPoolExecutor.State()
        workers = list(schedule.task_allocation.keys())
        num_nodes = len(list(schedule.task_graph.nodes()))
        with WorkerPool(workers) as executor:
            while len(state.results) != num_nodes:
                for task in state.eligible(schedule):
                    executor.submit(
                        schedule.get_processor(task.name),
                        execute_task,
                        [
                            state.results,
                            metered(return_meter=True)(task.payload[0]),
                            [
                                (
                                    task.inputs[arg].parent.name
                                    if arg in task.inputs
                                    else arg
                                )
                                for arg in task.payload[1]
                            ],
                            task.payload[2],
                        ],
                        callback=functools.partial(
                            ProcessPoolExecutor._on_task_complete,
                            state,
                            task.name,
                        ),
                        error_callback=functools.partial(
                            ProcessPoolExecutor._on_task_error, task.name
                        ),
                    )
                    state.pending.append(task.name)
                state.clear_cache(schedule.task_graph)
        return state

    def _execute_graph(graph: Graph, *, n_workers: int = 1):
        state = ProcessPoolExecutor.State()
        with mp.Pool(n_workers) as executor:
            while len(state.results) != len(list(graph.nodes())):
                for task in state.eligible(graph):
                    executor.apply_async(
                        execute_task,
                        [
                            state.results,
                            metered(return_meter=True)(task.payload[0]),
                            [
                                (
                                    task.inputs[arg].parent.name
                                    if arg in task.inputs
                                    else arg
                                )
                                for arg in task.payload[1]
                            ],
                            task.payload[2],
                        ],
                        callback=functools.partial(
                            ProcessPoolExecutor._on_task_complete,
                            state,
                            task.name,
                        ),
                        error_callback=functools.partial(
                            ProcessPoolExecutor._on_task_error, task.name
                        ),
                    )
                    state.pending.append(task.name)
            state.clear_cache(graph)
        return state

    def _execute(schedule: Schedule | Graph, **kwargs):
        if isinstance(schedule, Schedule):
            return ProcessPoolExecutor._execute_schedule(schedule, **kwargs)
        return ProcessPoolExecutor._execute_graph(schedule, **kwargs)

    def execute(schedule: Schedule | Graph, **kwargs) -> dict[str, Any]:
        state = ProcessPoolExecutor._execute(schedule, **kwargs)
        graph = schedule.task_graph if isinstance(schedule, Schedule) else schedule
        return {sink.name: state.results[sink.name] for sink in graph.sinks}

    def benchmark(schedule: Schedule | Graph, **kwargs) -> dict[str, Resources]:
        state = ProcessPoolExecutor._execute(schedule, **kwargs)
        return state.resources
