"""Integration test: submit a job to a gateway using SshCluster infra.

Spawns a gateway in-process, then submits job_noRuntime.py via an SshCluster
targeting the plain_cluster docker-compose environment (plain-cluster-main +
plain-cluster-worker1, plain-cluster-worker2).

Usage (from repo root):
    uv run python integration_tests/harnessCluster.py
"""

import logging
import sys
import time
from multiprocessing import Process
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from job_noRuntime import job  # ty:ignore[unresolved-import]

import cascade.gateway.api as api
import cascade.gateway.client as client
from cascade.gateway.__main__ import main_cli
from cascade.gateway.api import SshCluster
from cascade.ygg.transport import destroy_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
SSH_KEY = str(REPO_ROOT / "integration_tests" / "plain_cluster" / "ssh" / "id_ed25519")

GATEWAY_URL = "tcp://localhost:15355"
TRIES_LIMIT = 60
POLL_INTERVAL = 3.0


def spawn_gateway() -> Process:
    p = Process(
        target=main_cli,
        args=(GATEWAY_URL,),
        kwargs={"report_transport": "tcp"},
    )
    p.start()
    return p


def wait_for_gateway(url: str, retries: int = 20) -> None:
    """Poll until gateway responds or give up."""
    import zmq

    ctx = zmq.Context()
    for attempt in range(retries):
        try:
            s = ctx.socket(zmq.REQ)
            s.set(zmq.LINGER, 500)
            s.connect(url)
            s.send(b'{"clazz": "JobProgressRequest", "job_ids": [], "detailed_report": false}')
            ready = s.poll(800, zmq.POLLIN)
            s.close()
            if ready:
                logger.info(f"Gateway ready after {attempt + 1} attempt(s)")
                ctx.destroy()
                return
        except Exception:
            pass
        time.sleep(0.5)
    ctx.destroy()
    raise RuntimeError("Gateway did not become ready in time")


def main() -> None:
    destroy_context()
    gw = spawn_gateway()
    try:
        wait_for_gateway(GATEWAY_URL)

        ji = job()
        infra = SshCluster(
            controller_url="root@plain-cluster-main",
            worker_urls=["root@plain-cluster-worker1"],
            workers_per_host=1,
            ssh_key_path=SSH_KEY,
        )
        js = api.JobSpec(
            job_instance=ji,
            envvars={},
            infra_spec=infra,
        )

        logger.info("Submitting job to gateway...")
        submit_req = api.SubmitJobRequest(job=js)
        submit_res = client.request_response(submit_req, GATEWAY_URL, timeout_ms=5000)
        assert isinstance(submit_res, api.SubmitJobResponse), f"unexpected: {submit_res}"
        assert submit_res.error is None, f"submit error: {submit_res.error}"
        job_id = submit_res.job_id
        assert job_id is not None
        logger.info(f"Job submitted: {job_id}")

        # Poll until completion or failure
        for attempt in range(TRIES_LIMIT):
            prog_req = api.JobProgressRequest(job_ids=[job_id])
            prog_res = client.request_response(prog_req, GATEWAY_URL, timeout_ms=5000)
            assert isinstance(prog_res, api.JobProgressResponse)
            assert prog_res.error is None, f"progress error: {prog_res.error}"

            prog = prog_res.progresses.get(job_id)
            logger.info(f"[{attempt + 1}/{TRIES_LIMIT}] progress: {prog}")

            if prog is not None and hasattr(prog, "failure") and prog.failure is not None:  # ty:ignore[possibly-missing-attribute]
                raise RuntimeError(f"Job failed: {prog.failure}")  # ty:ignore[possibly-missing-attribute]

            if prog is not None and hasattr(prog, "pct") and prog.pct == "100.00":  # ty:ignore[possibly-missing-attribute]
                logger.info("Job completed successfully!")
                break

            time.sleep(POLL_INTERVAL)
        else:
            raise RuntimeError(f"Job did not complete within {TRIES_LIMIT * POLL_INTERVAL:.0f}s")

        # Retrieve output datasets
        datasets = prog_res.datasets.get(job_id, [])
        logger.info(f"Available datasets: {datasets}")
        for ds_id in datasets:
            result_req = api.ResultRetrievalRequest(job_id=job_id, dataset_id=ds_id)
            result_res = client.request_response(result_req, GATEWAY_URL, timeout_ms=5000)
            assert isinstance(result_res, api.ResultRetrievalResponse)
            if result_res.error:
                logger.warning(f"Could not retrieve {ds_id}: {result_res.error}")
            else:
                value = api.decoded_result(result_res, ji.jobInstance)
                logger.info(f"Result {ds_id} = {value!r}")

        # Shut down gateway
        shutdown_req = api.ShutdownRequest()
        shutdown_res = client.request_response(shutdown_req, GATEWAY_URL, timeout_ms=5000)
        assert hasattr(shutdown_res, "error") and shutdown_res.error is None
        gw.join(5)
        assert gw.exitcode == 0, f"gateway exited with {gw.exitcode}"
        logger.info("Gateway shut down cleanly.")

    except Exception:
        logger.exception("Integration test failed")
        if gw.is_alive():
            gw.kill()
        sys.exit(1)


if __name__ == "__main__":
    main()
