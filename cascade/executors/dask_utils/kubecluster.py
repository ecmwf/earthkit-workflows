import getpass
import os
import uuid

import kubernetes
import dask
from dask_kubernetes.classic import KubeCluster as DaskKubeCluster
from dask_kubernetes.classic.kubecluster import Pod, Worker, Scheduler
from dask_kubernetes.common.auth import ClusterAuth
from dask_kubernetes.common.objects import clean_pod_template
from dask_kubernetes.common.utils import (
    escape,
    get_current_namespace,
)
from dask_kubernetes.common.networking import get_scheduler_address


class KubeCluster(DaskKubeCluster):
    def __init__(
        self,
        pod_template=None,
        name=None,
        namespace=None,
        n_workers=None,
        host=None,
        port=None,
        env=None,
        auth=ClusterAuth.DEFAULT,
        idle_timeout=None,
        deploy_mode=None,
        interface=None,
        protocol=None,
        dashboard_address=None,
        security=None,
        scheduler_class=None,
        scheduler_kwargs=None,
        scheduler_service_wait_timeout=None,
        scheduler_service_name_resolution_retries=None,
        scheduler_pod_template=None,
        apply_default_affinity="preferred",
        **kwargs
    ):
        super().__init__(
            pod_template,
            name,
            namespace,
            n_workers,
            host,
            port,
            env,
            auth,
            idle_timeout,
            deploy_mode,
            interface,
            protocol,
            dashboard_address,
            security,
            scheduler_service_wait_timeout,
            scheduler_service_name_resolution_retries,
            scheduler_pod_template,
            apply_default_affinity,
            kwargs,
        )
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs

    async def _start(self):
        self.pod_template = self._get_pod_template(self.pod_template, pod_type="worker")
        self.scheduler_pod_template = self._get_pod_template(
            self.scheduler_pod_template, pod_type="scheduler"
        )
        if not self.pod_template:
            msg = (
                "Worker pod specification not provided. See KubeCluster "
                "docstring for ways to specify workers"
            )
            raise ValueError(msg)

        base_pod_template = self.pod_template
        self.pod_template = clean_pod_template(
            self.pod_template,
            apply_default_affinity=self.apply_default_affinity,
            pod_type="worker",
        )

        if not self.scheduler_pod_template:
            self.scheduler_pod_template = base_pod_template
        self.scheduler_pod_template.spec.containers[0].args = ["dask-scheduler"]

        self.scheduler_pod_template = clean_pod_template(
            self.scheduler_pod_template,
            apply_default_affinity=self.apply_default_affinity,
            pod_type="scheduler",
        )

        await ClusterAuth.load_first(self.auth)

        self.core_api = kubernetes.client.CoreV1Api()
        self.policy_api = kubernetes.client.PolicyV1Api()

        if self.namespace is None:
            self.namespace = get_current_namespace()

        environ = {k: v for k, v in os.environ.items() if k not in ["user", "uuid"]}
        self._generate_name = self._generate_name.format(
            user=getpass.getuser(), uuid=str(uuid.uuid4())[:10], **environ
        )
        self._generate_name = escape(self._generate_name)

        self.pod_template = self._fill_pod_templates(
            self.pod_template, pod_type="worker"
        )
        self.scheduler_pod_template = self._fill_pod_templates(
            self.scheduler_pod_template, pod_type="scheduler"
        )

        common_options = {
            "cluster": self,
            "core_api": self.core_api,
            "policy_api": self.policy_api,
            "namespace": self.namespace,
            "loop": self.loop,
        }

        if self._deploy_mode == "local":
            self.scheduler_spec = {
                "cls": dask.distributed.Scheduler
                if self.scheduler_class is None
                else self.scheduler_class,
                "options": {
                    "protocol": self._protocol,
                    "interface": self._interface,
                    "host": self.host,
                    "port": self.port,
                    "dashboard_address": self._dashboard_address,
                    "security": self.security,
                    **self.scheduler_kwargs,
                },
            }
        elif self._deploy_mode == "remote":
            self.scheduler_spec = {
                "cls": Scheduler
                if self.scheduler_class is None
                else self.scheduler_class,
                "options": {
                    "idle_timeout": self._idle_timeout,
                    "service_wait_timeout_s": self._scheduler_service_wait_timeout,
                    "service_name_retries": self._scheduler_service_name_resolution_retries,
                    "pod_template": self.scheduler_pod_template,
                    **common_options,
                    **self.scheduler_kwargs,
                },
            }
            assert isinstance(self.scheduler_spec["cls"], Pod)
        else:
            raise RuntimeError("Unknown deploy mode %s" % self._deploy_mode)

        self.new_spec = {
            "cls": Worker,
            "options": {"pod_template": self.pod_template, **common_options},
        }
        self.worker_spec = {i: self.new_spec for i in range(self._n_workers)}

        self.name = self.pod_template.metadata.generate_name

        await super()._start()

        if self._deploy_mode == "local":
            self.forwarded_dashboard_port = self.scheduler.services["dashboard"].port
        else:
            dashboard_address = await get_scheduler_address(
                self.scheduler.service.metadata.name,
                self.namespace,
                port_name="http-dashboard",
            )
            self.forwarded_dashboard_port = dashboard_address.split(":")[-1]
