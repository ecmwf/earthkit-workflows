# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The recv-loop of `gateway`, as well as basic deser. Actual business logic happens in `gateway.router`,
here we just match the right method of `gateway.router` based on what message we parsed
"""

import base64
import logging

import zmq

import cascade.gateway.api as api
from cascade.controller.report import deserialize
from cascade.executor.comms import get_context
from cascade.gateway.client import parse_request, serialize_response
from cascade.gateway.router import JobRouter

logger = logging.getLogger(__name__)


def handle_fe(socket: zmq.Socket, jobs: JobRouter) -> bool:
    rr = socket.recv()
    m = parse_request(rr)
    logger.debug(f"received frontend request {m}")
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
            progresses, datasets = jobs.progress_of(m.job_ids)
            rv = api.JobProgressResponse(
                progresses=progresses, datasets=datasets, error=None
            )
        except Exception as e:
            logger.exception(f"failed to get progress of: {m}")
            rv = api.JobProgressResponse(progresses={}, datasets={}, error=repr(e))
    elif isinstance(m, api.ResultRetrievalRequest):
        try:
            result = jobs.get_result(m.job_id, m.dataset_id)
            encoded = base64.b64encode(result)
            rv = api.ResultRetrievalResponse(result=encoded, error=None)
        except Exception as e:
            logger.exception(f"failed to get result: {m}")
            rv = api.ResultRetrievalResponse(result=None, error=repr(e))
    elif isinstance(m, api.ResultDeletionRequest):
        try:
            error = "\n".join(jobs.delete_results(m.datasets))
            rv = api.ResultDeletionResponse(error=error if error else None)
        except Exception as e:
            logger.exception(f"failed to get result: {m}")
            rv = api.ResultDeletionResponse(error=repr(e))
    elif isinstance(m, api.ShutdownRequest):
        jobs.shutdown()
        rv = api.ShutdownResponse(error=None)
    else:
        raise TypeError(m)
    response = serialize_response(rv)
    socket.send(response)
    return isinstance(rv, api.ShutdownResponse)


def handle_controller(socket: zmq.Socket, jobs: JobRouter) -> None:
    raw = socket.recv()
    report = deserialize(raw)
    logger.debug(f"received controller message {report}")
    jobs.maybe_update(report.job_id, report.current_status, report.timestamp)
    for dataset_id, result in report.results:
        jobs.put_result(report.job_id, dataset_id, result)


def serve(url: str, log_base: str | None = None) -> None:
    ctx = get_context()
    poller = zmq.Poller()

    fe = ctx.socket(zmq.REP)
    fe.bind(url)
    poller.register(fe, flags=zmq.POLLIN)
    jobs = JobRouter(poller, log_base)

    logger.debug("entering recv loop")
    is_break = False
    while not is_break:
        ready = poller.poll(None)
        for socket, _ in ready:
            if socket == fe:
                is_break = handle_fe(socket, jobs)
            else:
                handle_controller(socket, jobs)
