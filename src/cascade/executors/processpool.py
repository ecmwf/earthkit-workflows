import functools
import threading
from typing import Any

import multiprocess as mp
from meters import metered

from cascade.graph import Graph, Node
from cascade.schedulers.schedule import Schedule
from cascade.taskgraph import Resources
from cascade.utility import successors


class WorkerPool:
    """
    Process pool with named workers that allows submission of tasks to specific workers.
    Results from execution of submitted tasks are handled by a user specified callback on
    successful completion and an error callback on failure.
    """

    def __init__(self, workers: list[str]):
        self._inqueues = {name: mp.SimpleQueue() for name in workers}
        self._outqueue = mp.SimpleQueue()
        self._pool = [
            mp.Process(
                target=WorkerPool._worker, args=(self._inqueues[name], self._outqueue)
            )
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

    def _worker(inqueue: mp.SimpleQueue, outqueue: mp.SimpleQueue):
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

    def submit(
        self,
        worker: str,
        fn: callable,
        args: list,
        kwargs: dict | None = None,
        callback=None,
        error_callback=None,
    ):
        """
        Submit task to be run on a specific worker in the pool.

        Params
        ------
        worker: str, name of worker to run task
        fn: callable, function to be executed
        args: list, arguments to be passed to function
        kwargs: dict, keyword arguments to be passed to function
        callback: callable, function to be called on successful completion of task
        error_callback: callable, function to be called on failure of task

        Raises
        ------
        Exception: if WorkerPool is not running
        """
        if not self.is_running():
            raise Exception("WorkerPool result handler is not running")

        if kwargs is None:
            kwargs = {}
        self._inqueues[worker].put((fn, args, kwargs, callback, error_callback))

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

    def is_running(self) -> bool:
        """
        Returns True if result handler thread in WorkerPool is running, False otherwise.
        """
        return self.result_handler.is_alive()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()


class ProcessPoolExecutor:
    """
    Executor for executing and benchmarking graphs or schedules using a process pool.
    When executing a graph, the number of workers in the pool is determined by the
    n_workers parameter. When executing a schedule, the number of workers is determined
    by the number of processors in the schedule context graph.
    """

    class State:
        def __init__(self):
            self.pending = []
            self.results = mp.Manager().dict()
            self.resources = {}

        def eligible(self, graph: Graph) -> set[Node]:
            """
            Determines the next set of tasks that are ready for submission.

            Params
            ------
            graph: Graph or Schedule, task graph to be executed

            Returns
            -------
            set[Node]: set of tasks ready for submission
            """
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

        def clear_cache(self, graph: Graph):
            """
            Remove results from completed tasks which do not correspond to sinks
            and are no longer required by child tasks.

            Params
            ------
            graph: Graph or Schedule, task graph to be executed
            """
            tasks = list(self.results.keys())
            for task_name in tasks:
                if self.results[task_name] is not None:
                    task = graph.get_node(task_name)
                    if (
                        all([x.name in self.results for x in successors(graph, task)])
                        and task not in graph.sinks
                    ):
                        self.results[task_name] = None

    def __init__(self, n_workers: int = 1):
        self.state = None
        self.n_workers = n_workers

    def reset_state(self):
        self.state = ProcessPoolExecutor.State()

    def _on_task_complete(self, task_name: str, result: Any):
        """
        Callback function to be executed on successful completion of task, storing
        results and resources in the state.

        Params
        ------
        state: ProcessPoolExecutor.State, state of the executor
        task_name: str, name of task that was completed
        result: Any, result of the completed task
        """
        if task_name in self.state.results:
            raise ValueError(f"Task {task_name} already completed")
        meter, result = result
        self.state.results[task_name] = result
        # Convert memory to MB
        self.state.resources[task_name] = Resources(
            meter.elapsed_cpu, meter.mem / 10**6
        )

    def _on_task_error(task_name: str, error):
        """
        Callback function to be executed on failure of task.

        Params
        ------
        task_name: str, name of task that failed
        error: error returned by task

        Raises
        ------
        Exception: error returned by task
        """
        raise Exception(f"Error in task {task_name}: {error}")

    def _execute_task(
        results: dict[str, Any], function: callable, args: list, kwargs: dict
    ):
        extracted_args = [
            results[arg] if isinstance(arg, str) and arg in results else arg
            for arg in args
        ]
        return function(*extracted_args, **kwargs)

    def _execute_schedule(self, schedule: Schedule):
        """
        Execute task graph in schedule according to task allocation using WorkerPool.

        Params
        ------
        schedule: Schedule, task graph to be executed


        Raises
        ------
        Exception: if WorkerPool is not running
        """
        self.reset_state()
        workers = schedule.processors()
        num_nodes = len(list(schedule.nodes()))
        with WorkerPool(workers) as executor:
            while len(self.state.results) != num_nodes:
                for task in self.state.eligible(schedule):
                    executor.submit(
                        schedule.processor(task.name),
                        ProcessPoolExecutor._execute_task,
                        [
                            self.state.results,
                            metered(return_meter=True)(task.payload[0]),
                            [
                                (
                                    task.inputs[arg].parent.name
                                    if isinstance(arg, str) and arg in task.inputs
                                    else arg
                                )
                                for arg in task.payload[1]
                            ],
                            task.payload[2],
                        ],
                        callback=functools.partial(
                            self._on_task_complete,
                            task.name,
                        ),
                        error_callback=functools.partial(
                            ProcessPoolExecutor._on_task_error, task.name
                        ),
                    )
                    self.state.pending.append(task.name)
                self.state.clear_cache(schedule)

    def _execute_graph(self, graph: Graph):
        """
        Execute task graph using multiprocessing process pool.

        Params
        ------
        graph: Graph, task graph to be executed
        n_workers: int, number of workers in the process pool. Default is 1
        """
        self.reset_state()
        with mp.Pool(self.n_workers) as executor:
            while len(self.state.results) != len(list(graph.nodes())):
                for task in self.state.eligible(graph):
                    executor.apply_async(
                        ProcessPoolExecutor._execute_task,
                        [
                            self.state.results,
                            metered(return_meter=True)(task.payload[0]),
                            [
                                (
                                    task.inputs[arg].parent.name
                                    if isinstance(arg, str) and arg in task.inputs
                                    else arg
                                )
                                for arg in task.payload[1]
                            ],
                            task.payload[2],
                        ],
                        callback=functools.partial(
                            self._on_task_complete,
                            task.name,
                        ),
                        error_callback=functools.partial(
                            ProcessPoolExecutor._on_task_error, task.name
                        ),
                    )
                    self.state.pending.append(task.name)
                self.state.clear_cache(graph)

    def execute(self, graph: Graph) -> dict[str, Any]:
        """
        Execute graph or schedule using a pool of worker processes.

        Params
        ------
        graph: Graph or Schedule, task graph to be executed

        Returns
        -------
        dict[str, Any]: results of the sinks in the graph
        """
        if isinstance(graph, Schedule):
            self._execute_schedule(graph)
        else:
            self._execute_graph(graph)
        return {sink.name: self.state.results[sink.name] for sink in graph.sinks}
