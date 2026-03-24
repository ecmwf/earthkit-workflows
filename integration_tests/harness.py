from cascade.main import run_locally
import importlib
import sys
from base import JobSpec # ty:ignore[unresolved-import]
import logging

logger = logging.getLogger("cascade.benchmarks.harness")

if __name__ == "__main__":
    mod = importlib.import_module(sys.argv[1])
    job = mod.job()
    spc = mod.spc()

    rv = run_locally(job=job, hosts=spc.hosts, workers=spc.workers)
    logger.info(f"{rv.keys()=}")

    # TODO provide eg output file checking callbacks?
