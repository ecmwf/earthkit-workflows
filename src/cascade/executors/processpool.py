import multiprocess as mp
import functools
import threading
from typing import Any

from meters import metered
from cascade.utility import successors
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
            name="Result Handler",
            target=WorkerPool._handle_results,
            args=(self._outqueue,),
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

        handler_alive = self.result_handler.is_alive()
        self.result_handler.join()

        for p in self._pool:
            if p.is_alive():
                p.join()

        if not handler_alive:
            raise Exception("Result Hander thread died during execution")

    def is_running(self) -> bool:
        return self.result_handler.is_alive()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


class ProcessPoolExecutor:
    class State:
        def __init__(self):
            self.pending = []
            self.results = mp.Manager().dict()
            self.resources = {}

        def eligible(self, graph: Graph) -> set[Node]:
            if len(self.pending) == 0 and len(self.results) == 0:
                return set(graph.sources())

            # Determine next tasks ready for submission
            ready = set()
            index = 0
            while index < len(self.pending):
                task_name = self.pending[index]
                task = graph.get_node(task_name)
                if task_name in self.results:
                    next_tasks = successors(graph, task)
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
        workers = schedule.processors()
        num_nodes = len(list(schedule.nodes()))
        with WorkerPool(workers) as executor:
            while len(state.results) != num_nodes and executor.is_running():
                for task in state.eligible(schedule):
                    executor.submit(
                        schedule.processor(task.name),
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
                state.clear_cache(schedule)
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

    def _execute(graph: Graph, **kwargs):
        if isinstance(graph, Schedule):
            return ProcessPoolExecutor._execute_schedule(graph, **kwargs)
        return ProcessPoolExecutor._execute_graph(graph, **kwargs)

    def execute(graph: Graph, **kwargs) -> dict[str, Any]:
        state = ProcessPoolExecutor._execute(graph, **kwargs)
        return {sink.name: state.results[sink.name] for sink in graph.sinks}

    def benchmark(graph: Graph, **kwargs) -> dict[str, Resources]:
        state = ProcessPoolExecutor._execute(graph, **kwargs)
        return state.resources
