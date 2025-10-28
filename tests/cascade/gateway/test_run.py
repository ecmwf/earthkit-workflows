import time
from multiprocessing import Process

import cascade.gateway.api as api
import cascade.gateway.client as client
from cascade.gateway.__main__ import main
from cascade.low.builders import JobBuilder
from cascade.low.core import DatasetId, JobInstance, TaskDefinition, TaskInstance

init_value = 10
job_func = lambda i: i * 2
slow_func = lambda a: time.sleep(0.5) or a**2
tries_limit = 32


def get_job_succ() -> JobInstance:
    sod = TaskDefinition(
        func=TaskDefinition.func_enc(lambda: init_value),
        environment=[],
        input_schema={},
        output_schema=[("o", "int")],
    )
    soi = TaskInstance(definition=sod, static_input_kw={}, static_input_ps={})
    sid = TaskDefinition(
        func=TaskDefinition.func_enc(job_func),
        environment=[],
        input_schema={},  # TODO add 0: int once supported
        output_schema=[("o", "int")],
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


def get_job_fail() -> JobInstance:
    td = TaskDefinition(
        func=TaskDefinition.func_enc(job_func),
        environment=[],
        input_schema={},
        output_schema=[("o", "int")],
    )
    ti = TaskInstance(definition=td, static_input_kw={}, static_input_ps={"0": None})

    ji = JobBuilder().with_node("t", ti).build().get_or_raise()
    return ji


def get_job_slow() -> JobInstance:
    td = TaskDefinition(
        func=TaskDefinition.func_enc(slow_func),
        environment=[],
        input_schema={"a": "int"},
        output_schema=[("o", "int")],
    )
    ti = TaskInstance(definition=td, static_input_kw={"a": 4}, static_input_ps={})
    return JobBuilder().with_node("t", ti).build().get_or_raise()


def spawn_gateway(max_jobs: int | None = None) -> tuple[str, Process]:
    url = "tcp://localhost:12355"
    p = Process(target=main, args=(url,), kwargs={"max_jobs": max_jobs})
    p.start()
    return url, p


def test_job():
    url, gw = spawn_gateway(1)
    try:
        # succ job
        ji = get_job_succ()
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

        tries = 0
        job_progress_req = api.JobProgressRequest(job_ids=[job_id])
        while tries < tries_limit:
            job_progress_res = client.request_response(job_progress_req, url)
            assert job_progress_res.error is None
            is_computed = job_progress_res.progresses[job_id].pct == "100.00"
            is_datasets = ji.ext_outputs[0] in job_progress_res.datasets[job_id]
            if is_computed and is_datasets:
                break
            else:
                # NOTE not using logger, not properly configured in downstream-ci
                print(f"current progress is {job_progress_res}")
                tries += 1
                time.sleep(2)
        assert tries < tries_limit

        result_retrieval_req = api.ResultRetrievalRequest(
            job_id=job_id, dataset_id=ji.ext_outputs[0]
        )
        result_retrieval_res = client.request_response(result_retrieval_req, url)
        assert result_retrieval_res.error is None
        assert result_retrieval_res.result is not None
        deser = api.decoded_result(result_retrieval_res, ji)
        assert deser == job_func(init_value)

        result_deletion_req = api.ResultDeletionRequest(
            datasets={job_id: [ji.ext_outputs[0]]}
        )
        result_deletion_res = client.request_response(result_deletion_req, url)
        assert result_deletion_res.error is None

        # fail job
        ji = get_job_fail()
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

        tries = 0
        job_progress_req = api.JobProgressRequest(job_ids=[job_id])
        while tries < tries_limit:
            job_progress_res = client.request_response(job_progress_req, url)
            assert job_progress_res.error is None
            assert job_progress_res.progresses[job_id].pct != "100.00"
            if job_progress_res.progresses[job_id].failure is not None:
                break
            else:
                tries += 1
                time.sleep(2)
        assert tries < tries_limit

        # two jobs at once
        # TODO this is for manual test only, I dont have a good idea how to test, in a reliable & timely fashion,
        # that the execution has in effect been serial. But just running it and checking for success is still worth
        ji = get_job_slow()
        js = api.JobSpec(
            benchmark_name=None,
            envvars={},
            job_instance=ji,
            workers_per_host=1,
            hosts=1,
            use_slurm=False,
        )

        req = api.SubmitJobRequest(job=js)
        res1 = client.request_response(req, url)
        res2 = client.request_response(req, url)
        assert res1.error is None
        assert res2.error is None
        assert res1.job_id is not None
        assert res2.job_id is not None

        tries = 0
        job_ids = [res1.job_id, res2.job_id]
        job_progress_req = api.JobProgressRequest(job_ids=job_ids)
        while tries < tries_limit:
            job_progress_res = client.request_response(job_progress_req, url)
            assert job_progress_res.error is None
            is_computed = (
                lambda job_id: job_progress_res.progresses[job_id].pct == "100.00"
            )
            if all(is_computed(job_id) for job_id in job_ids):
                break
            else:
                # NOTE not using logger, not properly configured in downstream-ci
                print(f"current progress is {job_progress_res}")
                tries += 1
                time.sleep(2)
        assert tries < tries_limit

        # gw shutdown
        shutdown_req = api.ShutdownRequest()
        shutdown_res = client.request_response(shutdown_req, url, 3000)
        assert shutdown_res.error is None
        gw.join(5)
        assert gw.exitcode == 0

    except:
        if gw.is_alive():
            gw.kill()
        raise
