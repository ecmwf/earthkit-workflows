from typing import Any
import dask
from dask.highlevelgraph import HighLevelGraph
from dask.delayed import Delayed
from dask.distributed import Client, as_completed, performance_report

from cascade.transformers import to_dask_graph
from cascade.graph import Graph
from cascade.schedulers.schedule import Schedule

from .base import Executor
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
        adaptive_kwards: dict, arguments for making cluster adaptive (e.g. minimum, maximum)
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
            dask_graph = to_dask_graph(schedule.graph)

            # If task allocation specified then annotate graph and cluster with
            # worker names and priority according to the task allocation
            def _worker_name(key: str):
                return schedule.get_processor(key)

            def _priority(key: str):
                worker = schedule.get_processor(key)
                return len(schedule.task_allocation(worker)) - schedule.task_allocation(
                    worker
                ).index(key)

            # If cluster is adaptive then allow tasks to be scheduled in workers not specified in
            # original task allocation
            # TODO: check adaptive minimum number of workers is at least len(worker_names)
            with dask.annotate(
                workers=_worker_name,
                priority=_priority,
                allow_other_workers=(cluster._adaptive is not None),
            ):
                dask_graph = HighLevelGraph.from_collections("graph", dask_graph)
            sinks = schedule.graph.sinks
        else:
            assert isinstance(schedule, Graph)
            dask_graph = to_dask_graph(schedule)
            sinks = schedule.sinks

        cluster = create_cluster(cluster_type, cluster_kwargs, adaptive_kwargs)
        outputs = [Delayed(x.name, dask_graph) for x in sinks]

        # Set up distributed client
        dask.config.set(
            {"distributed.scheduler.worker-saturation": 1.0}
        )  # Important to prevent root task overloading
        if cluster is None:
            client = Client(
                n_workers=1,
                threads_per_worker=1,
                processes=True,
            )
        else:
            client = Client(cluster)

        with performance_report(report):
            future = client.compute(outputs)

            seq = as_completed(future)
            del future
            results = []
            # Trigger gargage collection on completed end tasks so scheduler doesn't
            # try to repeat them
            errored_tasks = 0
            for fut in seq:
                if fut.status != "finished":
                    print(f"Task failed with exception: {fut.exception()}")
                    errored_tasks += 1
                results.append(fut.result())

        client.close()

        if errored_tasks != 0:
            raise RuntimeError(f"{errored_tasks} task failed. Re-run required.")
        return results


class DaskLocalExecutor(Executor):
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
        memory_limit: str = "5G",
        cluster_kwargs: dict = {},
        adaptive_kwargs: dict = {},
        report: str = "performance_report.html",
    ) -> Any:
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


class DaskKubeExecutor(Executor):
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
        host: str = None,
        port: int = None,
        env: dict = None,
        cluster_kwargs: dict = {},
        adaptive_kwargs: dict = {},
        report: str = "performance_report.html",
    ) -> Any:
        return DaskExecutor.execute(
            schedule,
            cluster_type="kube",
            cluster_kwargs={
                "pod_template": pod_template,
                "namespace": namespace,
                "n_workers": n_workers,
                "host": host,
                "port": port,
                "env": env,
                **cluster_kwargs,
            },
            adaptive_kwargs=adaptive_kwargs,
            report=report,
        )
