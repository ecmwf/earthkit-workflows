"""
The recv-loop of `gateway`, as well as basic deser. Actual business logic happens in `gateway.router`,
here we just match the right method of `gateway.router` based on what message we parsed
"""

import zmq

from cascade.gateway.client import parse_request, serialize_response
import cascade.gateway.api as api
from cascade.gateway.router import JobRouter
from cascade.controller.report import JobProgress, deserialize
from cascade.executor.comms import get_context

def handle_fe(socket: zmq.Socket, jobs: JobRouter) -> None:
    rr = socket.recv()
    m = parse_request(rr)
    rv: api.CascadeGatewayAPI
    if isinstance(m, api.SubmitJobRequest):
        try:
            job_id = jobs.spawn_job(m.job)
            rv = api.SubmitJobResponse(job_id=job_id, error=None)
        except Exception as e:
            logger.exception(f"failed to spawn a job: {m}")
            rv = api.SubmitJobResponse(job_id=None, error=repr(e))
    elif isinstance(m, api.JobProgressRequest):
        try:
            progress = jobs.progress_of(m.job_id)
            rv = api.JobProgressResponse(progress=progress, error=None)
        except Exception as e:
            logger.exception(f"failed to get progress of: {m}")
            rv = api.JobProgressResponse(progress=None, error=repr(e))
    elif isinstance(m, api.ResultRetrievalRequest):
        try:
            result = jobs.get_result(m.job_id, m.dataset_id)
            rv = api.ResultRetrievalResponse(result=result, error=None)
        except Exception as e:
            logger.exception(f"failed to get result: {m}")
            rv = api.ResultRetrievalResponse(result=None, error=repr(e))
    else:
        raise TypeError(m)
    response = serialize_response(rv)
    socket.send(response)

def handle_controller(socket: zmq.Socket, jobs: JobRouter) -> None:
    raw = socket.recv()
    report = deserialize(raw)
    jobs.maybe_update(report.job_id, report.current_status, report.timestamp)
    for dataset_id, result in report.results:
        jobs.put_result(report.job_id, dataset_id, result)

def serve(url: str) -> None:
    cxt = get_context()
    sockets: list[zmq.Socket] = []

    fe = ctx.socket(zmq.REP)
    sockets.append(fe)
    jobs = JobRouter(sockets)

    while True:
        ready = zmq.poll(sockets, timeout=None)
        for socket in ready:
            if socket == fe:
                handle_fe(socket, jobs)
            else:
                handle_controller(socket, jobs)
