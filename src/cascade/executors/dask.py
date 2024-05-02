import functools
import pprint
from contextlib import nullcontext
from typing import Any

import dask
from dask.distributed import Client, as_completed, performance_report
from dask.graph_manipulation import chunks
from dask.highlevelgraph import HighLevelGraph

from cascade.contextgraph import ContextGraph
from cascade.executors.dask_utils import daskcontextgraph
from cascade.executors.executor import Executor
from cascade.graph import Graph
from cascade.schedulers.schedule import Schedule
from cascade.transformers import to_dask_graph

from .dask_utils import create_cluster


def execute(
    schedule: Graph | Schedule,
    client_kwargs: dict,
    adaptive: bool = False,
    report: str = "",
) -> Any:
    """
    Execute graph with a Dask cluster, e.g. LocalCluster or KubeCluster, and
    produce a performance report of the execution.

    Params
    ------
    schedule: Graph or Schedule, task graph to execute. If schedule is provided
    then annotates nodes with worker and priority according to the schedule
    client_kwargs: dict, arguments for Dask client
    adaptive: bool, whether cluster is adative or not
    report: str, name of performance report output file. If empty, no report is generated

    Returns
    -------
    Returns the outputs of the graph execution

    Raises
    ------
    RuntimeError if any tasks in the graph have failed
    """
    if isinstance(schedule, Schedule):
        # Functions for annotating tasks with workers and priority using task
        # allocation in schedule
        def _worker_name(task_allocation: dict, key: str):
            for processor, tasks in task_allocation.items():
                if key in tasks:
                    return processor

        def _priority(task_allocation: dict, key: str):
            for processor, tasks in task_allocation.items():
                if key in tasks:
                    worker = processor
                    break
            priority = len(task_allocation[worker]) - task_allocation[worker].index(key)
            return priority

        if adaptive:
            # If cluster is adaptive then allow tasks to be scheduled in workers not specified in
            # original task allocation
            with dask.annotate(
                workers=functools.partial(_worker_name, schedule.task_allocation),
                priority=functools.partial(_priority, schedule.task_allocation),
                allow_other_workers=True,
            ):
                dask_graph = HighLevelGraph.from_collections(
                    "graph", to_dask_graph(schedule)
                )
        else:
            # If strictly following schedule, need to modify tasks to bind with the previous
            # task in schedule task allocation
            with dask.annotate(
                workers=functools.partial(_worker_name, schedule.task_allocation),
            ):
                dask_graph = to_dask_graph(schedule)
                for _, allocation in schedule.task_allocation.items():
                    for index, task in enumerate(allocation):
                        if index > 0:
                            dask_graph[task] = (
                                chunks.bind,
                                dask_graph[task],
                                allocation[index - 1],
                            )
                dask_graph = HighLevelGraph.from_collections("graph", dask_graph)

    elif isinstance(schedule, Graph):
        dask_graph = to_dask_graph(schedule)
    else:
        raise ValueError("Invalid input, schedule must be of type Graph or Schedule")

    results = {}
    errored_tasks = 0
    context = nullcontext() if report == "" else performance_report(report)
    with Client(**client_kwargs) as client:
        with context:
            future = client.get(
                dask_graph, [x.name for x in schedule.sinks], sync=False
            )

            seq = as_completed(future)
            del future

            # Trigger gargage collection on completed end tasks so scheduler doesn't
            # try to repeat them

            for fut in seq:
                if fut.status != "finished":
                    print(f"Task {fut.key} failed with exception: {fut.exception()}")
                    errored_tasks += 1
                results[fut.key] = fut.result()

    if errored_tasks != 0:
        raise RuntimeError(f"{errored_tasks} tasks failed. Re-run required.")
    else:
        print("All tasks completed successfully.")
    return results


def create_context_graph(client_kwargs: dict) -> ContextGraph:
    with Client(**client_kwargs) as client:
        return daskcontextgraph.create_dask_context_graph(client)


def check_consistency(
    graph: Graph,
    cluster_kwargs: dict,
    adaptive_kwargs: dict | None = None,
):
    if not isinstance(graph, Schedule):
        return

    worker_names = list(graph.task_allocation.keys())
    cluster_kwargs["worker_names"] = worker_names
    cluster_kwargs["n_workers"] = len(worker_names)

    if adaptive_kwargs is not None:
        if adaptive_kwargs.get("minimum", len(worker_names)) < len(worker_names):
            raise ValueError(
                "Minimum number of workers in adaptive cluster must be at least processors in context graph"
            )
        elif adaptive_kwargs.get("maximum", len(worker_names)) < len(worker_names):
            raise ValueError(
                "Maximum number of workers in adaptive cluster must be at least processors in context graph"
            )


class DaskLocalExecutor(Executor):
    """
    Convenience class for DaskExecutor using LocalCluster exposing most
    common configuration arguments

    Options exposed:
        n_workers: int, number of Dask workers, default is 1. If a schedule is provided, this argument
        is overridden by the number of processors in schedule context graph
        threads_per_worker: int, number of threads per Dask worker
        processes: bool, whether to use processors (True) or threads (False). Defaults to True
        memory_limit: str, memory limit of each Dask worker. If None, no limit is applied
        cluster_kwargs: dict, arguments for Dask cluster
        adaptive_kwargs: dict, arguments for making cluster adaptive (e.g. minimum, maximum)
    """

    def __init__(
        self,
        n_workers: int = 1,
        threads_per_worker: int = 1,
        processes: bool = True,
        memory_limit: str | None = "5G",
        cluster_kwargs: dict = None,
        adaptive_kwargs: dict = None,
    ):
        self.adaptive_kwargs = adaptive_kwargs
        self.cluster_kwargs = {
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker,
            "processes": processes,
            "memory_limit": memory_limit,
        }
        if cluster_kwargs is not None:
            self.cluster_kwargs.update(cluster_kwargs)

    def execute(
        self,
        graph: Graph,
        *,
        report: str = "",
    ) -> Any:
        """
        Execute graph with a Dask local cluster and produce a performance report of the execution.

        Params
        ------
        graph: Graph or Schedule, task graph to execute. If schedule is provided
        then annotates nodes with worker and priority according to the schedule
        report: str, name of performance report output file. If empty, no report is generated

        Returns
        -------
        Returns output of graph execution in the form of dictionary containing sink name and
        corresponding output

        Raises
        ------
        RuntimeError if any tasks in the graph have failed
        """
        check_consistency(graph, self.cluster_kwargs, self.adaptive_kwargs)
        with create_cluster(
            "local", self.cluster_kwargs, self.adaptive_kwargs
        ) as cluster:
            pprint.pprint(cluster.scheduler_info)
            return execute(
                graph,
                client_kwargs={"address": cluster},
                adaptive=(self.adaptive_kwargs is not None),
                report=report,
            )

    def create_context_graph(self) -> ContextGraph:
        with create_cluster(
            "local", self.cluster_kwargs, self.adaptive_kwargs
        ) as cluster:
            return create_context_graph(client_kwargs={"address": cluster})


class DaskKubeExecutor(Executor):
    """
    Convenience class for DaskExecutor using KubeCluster exposing most
    common configuration arguments

    Options exposed:
        pod_template: dict, Kubernetes pod template. If None, defaults to using Dask docker image
        namespace: Kubernetes namespace in which to launch workers
        n_workers: int, number of Dask workers to launch. If a schedule is provided, this argument
        is overridden by the number of processors in schedule context graph
        env: dict, environment variables to pass to worker pods
        cluster_kwargs: dict, arguments for Dask cluster
        adaptive_kwargs: dict, arguments for making cluster adaptive (e.g. minimum, maximum)
    """

    def __init__(
        self,
        pod_template: dict = None,
        namespace: str = None,
        n_workers: int = None,
        env: dict = None,
        cluster_kwargs: dict = None,
        adaptive_kwargs: dict = None,
    ):
        self.cluster_kwargs = {
            "pod_template": pod_template,
            "namespace": namespace,
            "n_workers": n_workers,
            "env": env,
        }
        if cluster_kwargs is not None:
            self.cluster_kwargs.update(cluster_kwargs)
        self.adaptive_kwargs = adaptive_kwargs

    def execute(
        self,
        graph: Graph,
        *,
        report: str = "",
    ) -> Any:
        """
        Execute graph with a Dask Kubernetes cluster and produce a performance report of the execution.

        Params
        ------
        graph: Graph or Schedule, task graph to execute. If schedule is provided
        then annotates nodes with worker and priority according to the schedule
        report: str, name of performance report output file. If empty, no report is generated

        Returns
        -------
        Returns output of graph execution in the form of dictionary containing sink name and
        corresponding output

        Raises
        ------
        RuntimeError if any tasks in the graph have failed
        """
        check_consistency(graph, self.cluster_kwargs, self.adaptive_kwargs)
        with create_cluster(
            "kube", self.cluster_kwargs, self.adaptive_kwargs
        ) as cluster:
            pprint.pprint(cluster.scheduler_info)
            return execute(
                graph,
                client_kwargs={"address": cluster},
                adaptive=(self.adaptive_kwargs is not None),
                report=report,
            )

    def create_context_graph(self) -> ContextGraph:
        with create_cluster(
            "kube", self.cluster_kwargs, self.adaptive_kwargs
        ) as cluster:
            return create_context_graph(client_kwargs={"address": cluster})


class DaskClientExecutor(Executor):
    """
    Execute graph on existing Dask cluster, where the information for the scheduler is provided
    in the scheduler file
    """

    def __init__(self, dask_scheduler_file: str, adaptive: bool = False):
        self.dask_scheduler_file = dask_scheduler_file
        self.adaptive = adaptive

    def execute(
        self,
        graph: Graph,
        *,
        report: str = "",
    ) -> Any:
        """
        Execute graph on an existing dask cluster, where the information for the scheduler is provided
        in the scheduler file, and produce a performance report of the execution.

        Params
        ------
        graph: Graph or Schedule, task graph to execute. If schedule is provided
        then annotates nodes with worker and priority according to the schedule
        dask_scheduler_file: str, path to dask scheduler file
        adaptive: bool, whether cluster is adative or not
        report: str, name of performance report output file. If empty, no report is generated

        Returns
        -------
        Returns output of graph execution in the form of dictionary containing sink name and
        corresponding output

        Raises
        ------
        RuntimeError if any tasks in the graph have failed
        """
        return execute(
            graph,
            client_kwargs={"scheduler_file": self.dask_scheduler_file},
            adaptive=self.adaptive,
            report=report,
        )

    def create_context_graph(self) -> ContextGraph:
        return create_context_graph(
            client_kwargs={"scheduler_file": self.dask_scheduler_file}
        )
