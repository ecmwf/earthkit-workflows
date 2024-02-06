from typing import Any
import dask
from dask.highlevelgraph import HighLevelGraph
from dask.delayed import Delayed
from dask.distributed import Client, as_completed, performance_report
from dask.graph_manipulation import chunks
import functools

from cascade.transformers import to_dask_graph
from cascade.graph import Graph
from cascade.schedulers.schedule import Schedule

from .dask_utils import create_cluster


class DaskExecutor:
    def execute(
        schedule: Graph | Schedule,
        cluster_type: str = "local",
        cluster_kwargs: dict = {},
        adaptive_kwargs: dict = {},
        report: str = "performance_report.html",
    ) -> Any:
        """
        Execute graph with a Dask cluster, e.g. LocalCluster or KubeCluster, and
        produce a performance report of the execution.

        Params
        ------
        schedule: Graph or Schedule, task graph to execute. If schedule is provided
        then annotates nodes with worker and priority according to the schedule
        cluster_type: "local" or "kube", type of Dask cluster to execute the graph on.
        If not specified then defaults to default LocalCluster
        cluster_kwargs: dict, arguments for Dask cluster (e.g n_workers, threads_per_worker
        for LocalCluster)
        adaptive_kwargs: dict, arguments for making cluster adaptive (e.g. minimum, maximum)
        report: str, name of performance report output file

        Returns
        -------
        Returns the outputs of the graph execution

        Raises
        ------
        RuntimeError if any tasks in the graph have failed
        """

        if isinstance(schedule, Schedule):
            worker_names = list(schedule.task_allocation.keys())
            cluster_kwargs["worker_names"] = worker_names
            cluster_kwargs["n_workers"] = len(worker_names)

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
                priority = len(task_allocation[worker]) - task_allocation[worker].index(
                    key
                )
                return priority

            if len(adaptive_kwargs) > 0:
                assert adaptive_kwargs.get("minimum", len(worker_names)) >= len(
                    worker_names
                ), "Minimum number of workers in adaptive cluster must be at least processors in context graph"
                assert adaptive_kwargs.get("maximum", len(worker_names)) >= len(
                    worker_names
                ), "Maximum number of workers in adaptive cluster must be at least processors in context graph"
                # If cluster is adaptive then allow tasks to be scheduled in workers not specified in
                # original task allocation
                with dask.annotate(
                    workers=functools.partial(_worker_name, schedule.task_allocation),
                    priority=functools.partial(_priority, schedule.task_allocation),
                    allow_other_workers=True,
                ):
                    dask_graph = HighLevelGraph.from_collections(
                        "graph", to_dask_graph(schedule.task_graph)
                    )
            else:
                # If strictly following schedule, need to modify tasks to bind with the previous
                # task in schedule task allocation
                with dask.annotate(
                    workers=functools.partial(_worker_name, schedule.task_allocation),
                ):
                    dask_graph = to_dask_graph(schedule.task_graph)
                    for _, allocation in schedule.task_allocation.items():
                        for index, task in enumerate(allocation):
                            if index > 0:
                                dask_graph[task] = (
                                    chunks.bind,
                                    dask_graph[task],
                                    allocation[index - 1],
                                )
                    dask_graph = HighLevelGraph.from_collections("graph", dask_graph)

            outputs = [x.name for x in schedule.task_graph.sinks]
        else:
            assert isinstance(schedule, Graph)
            dask_graph = to_dask_graph(schedule)
            outputs = [x.name for x in schedule.sinks]

        cluster = create_cluster(cluster_type, cluster_kwargs, adaptive_kwargs)

        # Set up distributed client
        dask.config.set(
            {"distributed.scheduler.worker-saturation": 1.0}
        )  # Important to prevent root task overloading
        client = Client(cluster)

        with performance_report(report):
            future = client.get(dask_graph, outputs, sync=False)

            seq = as_completed(future)
            del future
            results = {}
            # Trigger gargage collection on completed end tasks so scheduler doesn't
            # try to repeat them
            errored_tasks = 0
            for fut in seq:
                if fut.status != "finished":
                    print(f"Task {fut.key} failed with exception: {fut.exception()}")
                    errored_tasks += 1
                assert fut.key not in results
                results[fut.key] = fut.result()

        client.close()

        if errored_tasks != 0:
            raise RuntimeError(f"{errored_tasks} task failed. Re-run required.")
        return results


class DaskLocalExecutor:
    """
    Convenience class for DaskExecutor using LocalCluster exposing most
    common configuration arguments
    """

    @classmethod
    def execute(
        cls,
        schedule: Graph | Schedule,
        n_workers: int = 1,
        threads_per_worker: int = 1,
        processes: bool = True,
        memory_limit: str | None = "5G",
        cluster_kwargs: dict = {},
        adaptive_kwargs: dict = {},
        report: str = "performance_report.html",
    ) -> Any:
        """
        Execute graph with a Dask local cluster and produce a performance report of the execution.

        Params
        ------
        schedule: Graph or Schedule, task graph to execute. If schedule is provided
        then annotates nodes with worker and priority according to the schedule
        n_workers: int, number of Dask workers, default is 1. If a schedule is provided, this argument
        is overridden by the number of processors in schedule context graph
        threads_per_worker: int, number of threads per Dask worker
        processes: bool, whether to use processors (True) or threads (False). Defaults to True
        memory_limit: str, memory limit of each Dask worker. If None, no limit is applied
        cluster_kwargs: dict, arguments for Dask cluster
        adaptive_kwargs: dict, arguments for making cluster adaptive (e.g. minimum, maximum)
        report: str, name of performance report output file

        Returns
        -------
        Returns output of graph execution in the form of dictionary containing sink name and
        corresponding output

        Raises
        ------
        RuntimeError if any tasks in the graph have failed
        """
        return DaskExecutor.execute(
            schedule,
            cluster_type="local",
            cluster_kwargs={
                "n_workers": n_workers,
                "threads_per_worker": threads_per_worker,
                "processes": processes,
                "memory_limit": memory_limit,
                **cluster_kwargs,
            },
            adaptive_kwargs=adaptive_kwargs,
            report=report,
        )


class DaskKubeExecutor:
    """
    Convenience class for DaskExecutor using KubeCluster exposing most
    common configuration arguments
    """

    @classmethod
    def execute(
        cls,
        schedule: Graph | Schedule,
        pod_template: dict = None,
        namespace: str = None,
        n_workers: int = None,
        env: dict = None,
        cluster_kwargs: dict = {},
        adaptive_kwargs: dict = {},
        report: str = "performance_report.html",
    ) -> Any:
        """
        Execute graph with a Dask Kubernetes cluster and produce a performance report of the execution.

        Params
        ------
        schedule: Graph or Schedule, task graph to execute. If schedule is provided
        then annotates nodes with worker and priority according to the schedule
        pod_template: dict, Kubernetes pod template. If None, defaults to using Dask docker image
        namespace: Kubernetes namespace in which to launch workers
        n_workers: int, number of Dask workers to launch. If a schedule is provided, this argument
        is overridden by the number of processors in schedule context graph
        env: dict, environment variables to pass to worker pods
        cluster_kwargs: dict, arguments for Dask cluster
        adaptive_kwargs: dict, arguments for making cluster adaptive (e.g. minimum, maximum)
        report: str, name of performance report output file

        Returns
        -------
        Returns output of graph execution in the form of dictionary containing sink name and
        corresponding output

        Raises
        ------
        RuntimeError if any tasks in the graph have failed
        """
        return DaskExecutor.execute(
            schedule,
            cluster_type="kube",
            cluster_kwargs={
                "pod_template": pod_template,
                "namespace": namespace,
                "n_workers": n_workers,
                "env": env,
                **cluster_kwargs,
            },
            adaptive_kwargs=adaptive_kwargs,
            report=report,
        )
