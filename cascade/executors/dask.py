from typing import Any
import dask
from dask.delayed import Delayed
from dask.distributed import SpecCluster, Client, as_completed, performance_report

from .base import Executor
from cascade.transformers import to_dask_graph
from cascade.graph import Graph


class DaskExecutor(Executor):
    def execute(
        graph: Graph,
        *,
        cluster: SpecCluster | None = None,
        report: str = "performance_report.html",
    ) -> Any:
        """
        Execute graph with a Dask cluster, e.g. LocalCluster or KubeCluster and
        produce a performance report of the execution.

        Params
        ------
        graph: Graph, task graph to execute
        cluster: SpecCluster, Dask cluster to execute the graph on. If not specified
        then defaults to default LocalCluster
        report: str, name of performance report output file

        Returns
        -------
        Returns the outputs of the graph execution

        Raises
        ------
        RuntimeError if any tasks in the graph have failed
        """
        dask_graph = to_dask_graph(graph)
        outputs = [Delayed(x.name, dask_graph) for x in graph.sinks]

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
