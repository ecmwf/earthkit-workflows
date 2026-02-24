# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Contains utility methods for benchmark definitions"""

import logging
import logging.config
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns
from typing import Any

import orjson

import cascade.executor.platform as platform
from cascade.controller.impl import run
from cascade.executor.bridge import Bridge
from cascade.executor.comms import callback
from cascade.executor.config import logging_config, logging_config_filehandler
from cascade.executor.executor import Executor
from cascade.executor.msg import BackboneAddress, ExecutorShutdown
from cascade.low.core import DatasetId, JobInstance, JobInstanceRich
from cascade.low.func import msum
from cascade.scheduler.precompute import precompute

logger = logging.getLogger("cascade.benchmarks")


def get_job(benchmark: str | None, instance_path: str | None) -> JobInstanceRich:
    # NOTE we dont want to import these at the top level to prevent imports pollution of executor
    import cascade.low.into
    from earthkit.workflows.graph import Graph, deduplicate_nodes
    # NOTE because of os.environ, we don't import all... ideally we'd have some file-based init/config mech instead
    if benchmark is not None and instance_path is not None:
        raise TypeError("specified both benchmark name and job instance")
    elif instance_path is not None:
        with open(instance_path, "rb") as f:
            d = orjson.loads(f.read())
            return JobInstanceRich(**d)
    elif benchmark is not None:
        instance: JobInstance
        if benchmark.startswith("j1"):
            import benchmarks_old.job1 as job1

            graphs = {
                "j1.prob": job1.get_prob(),
                "j1.ensms": job1.get_ensms(),
                "j1.efi": job1.get_efi(),
            }
            union = lambda prefix: deduplicate_nodes(
                msum((v for k, v in graphs.items() if k.startswith(prefix)), Graph)
            )
            graphs["j1.all"] = union("j1.")
            instance = cascade.low.into.graph2job(graphs[benchmark])
        elif benchmark.startswith("generators"):
            import benchmarks_old.generators as generators

            instance = generators.get_job()
        elif benchmark.startswith("matmul"):
            import benchmarks_old.matmul as matmul

            instance = matmul.get_job()
        elif benchmark.startswith("dist"):
            import benchmarks_old.dist as dist

            instance = dist.get_job()
        elif benchmark.startswith("dask"):
            import benchmarks_old.dask as dask

            instance = dask.get_job(benchmark[len("dask.") :])
        else:
            raise NotImplementedError(benchmark)
        return JobInstanceRich(jobInstance=instance, checkpointSpec=None)
    else:
        raise TypeError("specified neither benchmark name nor job instance")
