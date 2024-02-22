from dask.distributed import SpecCluster


def create_cluster(
    type: str, cluster_kwargs: dict, adaptive_kwargs: dict | None = None
) -> SpecCluster:
    if type == "local":
        from .localcuster import LocalCluster

        cluster = LocalCluster(**cluster_kwargs)
    elif type == "kube":
        from .kubecluster import KubeCluster

        cluster = KubeCluster(**cluster_kwargs)

    if adaptive_kwargs is not None:
        cluster.adapt(**adaptive_kwargs)
    return cluster
