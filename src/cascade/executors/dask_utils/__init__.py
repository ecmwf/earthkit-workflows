from dask.distributed import SpecCluster
import dask


def create_cluster(
    type: str, cluster_kwargs: dict, adaptive_kwargs: dict | None = None
) -> SpecCluster:
    dask.config.set(
        {
            "distributed.scheduler.worker-saturation": 1.0,
            "distributed.scheduler.worker-ttl": "20 minutes",
        }
    )  # Important to prevent root task overloading

    if type == "local":
        from .localcuster import LocalCluster

        cluster = LocalCluster(**cluster_kwargs)
    elif type == "kube":
        from .kubecluster import KubeCluster

        cluster = KubeCluster(**cluster_kwargs)

    if adaptive_kwargs is not None:
        cluster.adapt(**adaptive_kwargs)
    return cluster
