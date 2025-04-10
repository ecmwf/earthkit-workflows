import time
from multiprocessing import Process

import cascade.gateway.api as api
import cascade.gateway.client as client
from cascade.gateway.__main__ import main as gateway_entrypoint
from cascade.low.builders import JobBuilder
from cascade.low.core import DatasetId, JobInstance, TaskDefinition, TaskInstance

init_value = 10
job_func = lambda i: i * 2


def get_job() -> JobInstance:
    sod = TaskDefinition(
        func=TaskDefinition.func_enc(lambda: init_value),
        environment=[],
        input_schema={},
        output_schema={"o": "int"},
    )
    soi = TaskInstance(definition=sod, static_input_kw={}, static_input_ps={})
    sid = TaskDefinition(
        func=TaskDefinition.func_enc(job_func),
        environment=[],
        input_schema={},  # TODO add 0: int once supported
        output_schema={"o": "int"},
    )
    sii = TaskInstance(definition=sid, static_input_kw={}, static_input_ps={})

    ji = (
        JobBuilder()
        .with_node("so", soi)
        .with_node("si", sii)
        .with_edge("so", "si", 0, "o")
        .build()
        .get_or_raise()
    )
    ji.ext_outputs = [DatasetId("si", "o")]
    return ji


def spawn_gateway() -> tuple[str, Process]:
    url = "tcp://localhost:12355"
    p = Process(target=gateway_entrypoint, args=(url,))
    p.start()
    return url, p


def test_job():
    url, gw = spawn_gateway()
    try:
        ji = get_job()
        js = api.JobSpec(
            benchmark_name=None,
            envvars={},
            job_instance=ji,
            workers_per_host=1,
            hosts=1,
            use_slurm=False,
        )

        submit_job_req = api.SubmitJobRequest(job=js)
        submit_job_res = client.request_response(submit_job_req, url)
        job_id = submit_job_res.job_id
        assert submit_job_res.error is None
        assert job_id is not None

        tries = 3
        job_progress_req = api.JobProgressRequest(job_ids=[job_id])
        while tries > 0:
            job_progress_res = client.request_response(job_progress_req, url)
            assert job_progress_res.error is None
            if job_progress_res.progresses[job_id] == "100.00":
                break
            else:
                tries -= 1
                time.sleep(1)
        assert tries > 0

        result_retrieval_req = api.ResultRetrievalRequest(
            job_id=job_id, dataset_id=ji.ext_outputs[0]
        )
        result_retrieval_res = client.request_response(result_retrieval_req, url)
        assert result_retrieval_res.error is None
        assert result_retrieval_res.result is not None
        deser = api.decoded_result(result_retrieval_res, ji)
        assert deser == job_func(init_value)

        shutdown_req = api.ShutdownRequest()
        shutdown_res = client.request_response(shutdown_req, url)
        assert shutdown_res.error is None
        gw.join(5)
        assert gw.exitcode == 0
    except:
        if gw.is_alive():
            gw.kill()
        raise
