import logging
import warnings
import toolz
import math


from dask.distributed import LocalCluster as DaskLocalCluster
from dask.distributed import Worker, Nanny, Scheduler, SpecCluster, Security
from dask.system import CPU_COUNT
from dask.distributed.deploy.utils import nprocesses_nthreads
from dask.distributed.worker_memory import parse_memory_limit

logger = logging.getLogger("local")


class LocalCluster(DaskLocalCluster):
    def __init__(
        self,
        name=None,
        n_workers=None,
        threads_per_worker=None,
        processes=True,
        loop=None,
        start=None,
        host=None,
        ip=None,
        scheduler_port=0,
        silence_logs=logging.WARN,
        dashboard_address=":8787",
        worker_dashboard_address=None,
        diagnostics_port=None,
        services=None,
        worker_services=None,
        service_kwargs=None,
        asynchronous=False,
        security=None,
        protocol=None,
        blocked_handlers=None,
        interface=None,
        worker_class=None,
        scheduler_kwargs=None,
        scheduler_sync_interval=1,
        worker_names=None,
        **worker_kwargs,
    ):
        if ip is not None:
            # In the future we should warn users about this move
            # warnings.warn("The ip keyword has been moved to host")
            host = ip

        if diagnostics_port is not None:
            warnings.warn(
                "diagnostics_port has been deprecated. "
                "Please use `dashboard_address=` instead"
            )
            dashboard_address = diagnostics_port

        if threads_per_worker == 0:
            warnings.warn(
                "Setting `threads_per_worker` to 0 has been deprecated. "
                "Please set to None or to a specific int."
            )
            threads_per_worker = None

        if "dashboard" in worker_kwargs:
            warnings.warn(
                "Setting `dashboard` is discouraged. "
                "Please set `dashboard_address` to affect the scheduler (more common) "
                "and `worker_dashboard_address` for the worker (less common)."
            )

        if processes is None:
            processes = worker_class is None or issubclass(worker_class, Nanny)
        if worker_class is None:
            worker_class = Nanny if processes else Worker

        self.status = None
        self.processes = processes

        if security is None:
            # Falsey values load the default configuration
            security = Security()
        elif security is True:
            # True indicates self-signed temporary credentials should be used
            security = Security.temporary()
        elif not isinstance(security, Security):
            raise TypeError("security must be a Security object")

        if protocol is None:
            if host and "://" in host:
                protocol = host.split("://")[0]
            elif security and security.require_encryption:
                protocol = "tls://"
            elif not self.processes and not scheduler_port:
                protocol = "inproc://"
            else:
                protocol = "tcp://"
        if not protocol.endswith("://"):
            protocol = protocol + "://"

        if host is None and not protocol.startswith("inproc") and not interface:
            host = "127.0.0.1"

        services = services or {}
        worker_services = worker_services or {}
        if n_workers is None and threads_per_worker is None:
            if processes:
                n_workers, threads_per_worker = nprocesses_nthreads()
            else:
                n_workers = 1
                threads_per_worker = CPU_COUNT
        if n_workers is None and threads_per_worker is not None:
            n_workers = max(1, CPU_COUNT // threads_per_worker) if processes else 1
        if n_workers and threads_per_worker is None:
            # Overcommit threads per worker, rather than undercommit
            threads_per_worker = max(1, int(math.ceil(CPU_COUNT / n_workers)))
        if n_workers and "memory_limit" not in worker_kwargs:
            worker_kwargs["memory_limit"] = parse_memory_limit(
                "auto", 1, n_workers, logger=logger
            )

        worker_kwargs.update(
            {
                "host": host,
                "nthreads": threads_per_worker,
                "services": worker_services,
                "dashboard_address": worker_dashboard_address,
                "dashboard": worker_dashboard_address is not None,
                "interface": interface,
                "protocol": protocol,
                "security": security,
                "silence_logs": silence_logs,
            }
        )

        scheduler = {
            "cls": Scheduler,
            "options": toolz.merge(
                dict(
                    host=host,
                    services=services,
                    service_kwargs=service_kwargs,
                    security=security,
                    port=scheduler_port,
                    interface=interface,
                    protocol=protocol,
                    dashboard=dashboard_address is not None,
                    dashboard_address=dashboard_address,
                    blocked_handlers=blocked_handlers,
                ),
                scheduler_kwargs or {},
            ),
        }

        self.worker_names = worker_names
        worker = {"cls": worker_class, "options": worker_kwargs}
        workers = {self._new_worker_name(i): worker for i in range(n_workers)}

        SpecCluster.__init__(
            self,
            name=name,
            scheduler=scheduler,
            workers=workers,
            worker=worker,
            loop=loop,
            asynchronous=asynchronous,
            silence_logs=silence_logs,
            security=security,
            scheduler_sync_interval=scheduler_sync_interval,
        )

    def _new_worker_name(self, worker_number):
        # Revert to default worker names if worker_number
        # greater than provided set of names for cases of adaptive scaling
        if self.worker_names is None or worker_number >= len(worker_number):
            return super()._new_worker_name(worker_number)
        return self.worker_names[worker_number]

    def scale(self, n=0, memory=None, cores=None):
        if self.worker_names is not None:
            assert n >= len(
                self.work_names
            ), f"Can not scale down cluster to less than {len(self.worker_names)}\
with static scheduling enabled. Scheduler will hang waiting to schedule job on specified workers"
        super().scale(n, memory, cores)
