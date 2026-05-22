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
import datetime as dt
import logging
import os

import zmq

import cascade.executor.platform as platform
import cascade.gateway.api as api
from cascade.controller.report import deserialize
from cascade.deployment.logging import LoggingConfig, init_from_obj
from cascade.gateway.client import parse_request, serialize_response
from cascade.gateway.router import JobRouter
from cascade.low.exceptions import CascadeInternalError
from cascade.ygg.api import YggNode

logger = logging.getLogger(__name__)


def handle_fe(socket: zmq.Socket, jobs: JobRouter) -> bool:
    rr = socket.recv()
    m = parse_request(rr)
    logger.debug(f"received frontend request {m}")
    rv: api.CascadeGatewayAPI
    if isinstance(m, api.SubmitJobRequest):
        try:
            job_id, error = jobs.enqueue_job(m.job)
            rv = api.SubmitJobResponse(job_id=job_id, error=error)
        except Exception as e:
            logger.exception(f"failed to spawn a job: {m}")
            rv = api.SubmitJobResponse(job_id=None, error=repr(e))
    elif isinstance(m, api.JobProgressRequest):
        try:
            rv = jobs.progress_of(m.job_ids, m.detailed_report)
        except Exception as e:
            logger.exception(f"failed to get progress of: {m}")
            rv = api.JobProgressResponse(progresses={}, datasets={}, error=repr(e), queue_length=-1)
    elif isinstance(m, api.ResultRetrievalRequest):
        try:
            result, error = jobs.get_result(m.job_id, m.dataset_id)
            if error is not None:
                rv = api.ResultRetrievalResponse(result=None, error=error)
            elif result is None:
                rv = api.ResultRetrievalResponse(result=None, error="unexpected empty result")
            else:
                encoded = base64.b64encode(result).decode("ascii")
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
        raise CascadeInternalError(f"unexpected message type in gateway handle_fe: {type(m)}")
    response = serialize_response(rv)
    socket.send(response)
    return isinstance(rv, api.ShutdownResponse)


def handle_controller(ygg: YggNode, jobs: JobRouter) -> None:
    while msgs := ygg.poll_messages(timeout_ms=0):
        for msg in msgs:
            raw_report = msg.payload
            report = deserialize(raw_report)
            logger.debug(f"received controller message {report}")
            for dataset_id, result in report.results:
                jobs.put_result(report.job_id, dataset_id, result)
            jobs.maybe_update(report.job_id, report.current_status, report.timestamp, report.completed_task, report.planned_tasks)


def serve(
    url: str,
    loggingConfig: LoggingConfig,
    troika_config: str | None = None,
    max_concurrent_jobs: int | None = None,
    max_jobs_history: int = 20,
    max_queue_length: int = 50,
    report_transport: str = "tcp",
) -> None:
    if report_transport == "tcp":
        bind_base = f"tcp://{platform.get_bindabble_self()}"
        ygg = YggNode(f"{bind_base}:*")
    elif report_transport == "ipc":
        bind = f"ipc:///tmp/gateway.{os.getpid()}.socket"
        ygg = YggNode(bind)
    else:
        raise NotImplementedError(report_transport)
    poller = zmq.Poller()

    fe = zmq.Context().socket(zmq.REP)
    fe.bind(url)
    # TODO migrate fe socket to ygg once REP socket type supported
    ygg_control_socket = ygg._listener._socket_by_lane["control"]  # ty: ignore
    poller.register(fe, flags=zmq.POLLIN)
    poller.register(ygg_control_socket, flags=zmq.POLLIN)
    jobs = JobRouter(ygg, loggingConfig, troika_config, max_concurrent_jobs, max_jobs_history, max_queue_length)

    logger.debug("entering recv loop")
    is_break = False
    try:
        while not is_break:
            ready = poller.poll(None)
            for socket, _ in ready:
                if socket == fe:
                    is_break = handle_fe(socket, jobs)
                elif socket == ygg_control_socket:
                    handle_controller(ygg, jobs)
                else:
                    raise CascadeInternalError(f"unexpected socket in gateway poller loop: {socket=}")
    finally:
        fe.close()
        ygg.close()


def roleLoggingStr() -> str:
    # In case there are multiple gateway restarts etc, we dont want to collide.
    # For other roles this matters not -- they are distinguished by jobId in the name, or by #attempt counter
    now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"gateway.{now}"


def main_enp(
    url: str,
    loggingConfig: LoggingConfig,
    max_concurrent_jobs: int | None,
    max_jobs_history: int = 20,
    max_queue_length: int = 50,
    report_transport: str = "tcp",
) -> None:
    # use when process is not __main__ but eg forked from another
    init_from_obj(loggingConfig, roleLoggingStr())
    serve(url, loggingConfig, None, max_concurrent_jobs, max_jobs_history, max_queue_length, report_transport)
