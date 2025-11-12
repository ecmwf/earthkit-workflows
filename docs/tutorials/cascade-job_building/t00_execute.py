"""Simplified interface for job submission"""

import random
from typing import Any

from cascade.benchmarks.util import run_locally
from cascade.low.core import DatasetId, JobInstance


def run_job(job: JobInstance) -> dict[DatasetId, Any]:
    return run_locally(
        job,
        hosts=1,
        workers=3,
        portBase=12345 + random.randint(0, 100),
    )
