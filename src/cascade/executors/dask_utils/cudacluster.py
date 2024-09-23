import logging

from dask_cuda import LocalCUDACluster

logger = logging.getLogger("local")


class CudaCluster(LocalCUDACluster): ...
