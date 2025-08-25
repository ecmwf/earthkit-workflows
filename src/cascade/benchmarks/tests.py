# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Module for simplifying writing tests

Similar to util, but not enough to unify

It is capable, for a single given task, to spin an shm server, put all task's inputs into it, execute the task, store outputs in memory, and retrieve the result.
See the `demo()` function at the very end
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable

import cloudpickle

import cascade.executor.platform as platform
import cascade.shm.api as shm_api
import cascade.shm.client as shm_client
from cascade.executor.comms import Listener as ZmqListener
from cascade.executor.config import logging_config
from cascade.executor.msg import BackboneAddress, DatasetPublished
from cascade.executor.runner.memory import Memory, ds2shmid
from cascade.executor.runner.packages import PackagesEnv
from cascade.executor.runner.runner import ExecutionContext, run
from cascade.low.builders import TaskBuilder
from cascade.low.core import DatasetId
from cascade.shm.server import entrypoint as shm_server

logger = logging.getLogger(__name__)


@contextmanager
def setup_shm(testId: str):
    mp_ctx = platform.get_mp_ctx("executor-aux")
    shm_socket = f"/tmp/tcShm-{testId}"
    shm_api.publish_socket_addr(shm_socket)
    shm_process = mp_ctx.Process(
        target=shm_server,
        kwargs={
            "logging_config": logging_config,
            "shm_pref": f"tc{testId}",
        },
    )
    shm_process.start()
    shm_client.ensure()
    try:
        yield
    except Exception as e:
        # NOTE we log like this in case shm shutdown freezes
        logger.exception(f"gotten {repr(e)}, proceed with shm shutdown")
        raise
    finally:
        shm_client.shutdown(timeout_sec=1.0)
        shm_process.join(1)
        if shm_process.is_alive():
            shm_process.terminate()
            shm_process.join(1)
        if shm_process.is_alive():
            shm_process.kill()
            shm_process.join(1)


def simple_runner(callback: BackboneAddress, executionContext: ExecutionContext):
    tasks = list(executionContext.tasks.keys())
    if len(tasks) != 1:
        raise ValueError(f"expected 1 task, gotten {len(tasks)}")
    taskId = tasks[0]
    taskInstance = executionContext.tasks[taskId]
    with Memory(callback, "testWorker") as memory, PackagesEnv() as pckg:
        # for key, value in taskSequence.extra_env.items():
        #    os.environ[key] = value

        pckg.extend(taskInstance.definition.environment)
        run(taskId, executionContext, memory)
        memory.flush()


@dataclass
class CallableInstance:
    func: Callable
    kwargs: dict[str, Any]
    args: list[tuple[int, Any]]
    env: list[str]
    exp_output: Any


def callable2ctx(
    callableInstance: CallableInstance, callback: BackboneAddress
) -> ExecutionContext:
    taskInstance = TaskBuilder.from_callable(
        callableInstance.func, callableInstance.env
    )
    param_source = {}
    params = [
        (key, DatasetId("taskId", f"kwarg.{key}"), value)
        for key, value in callableInstance.kwargs.items()
    ] + [
        (key, DatasetId("taskId", f"pos.{key}"), value)
        for key, value in callableInstance.args
    ]
    for key, ds_key, value in params:
        raw = cloudpickle.dumps(value)
        L = len(raw)
        buf = shm_client.allocate(ds2shmid(ds_key), L, "cloudpickle.loads")
        buf.view()[:L] = raw
        buf.close()
        param_source[key] = (ds_key, "Any")

    return ExecutionContext(
        tasks={"taskId": taskInstance},
        param_source={"taskId": param_source},
        callback=callback,
        publish={
            DatasetId("taskId", output)
            for output, _ in taskInstance.definition.output_schema
        },
    )


def run_test(
    callableInstance: CallableInstance, testId: str, max_runtime_sec: int
) -> Any:
    with setup_shm(testId):
        addr = f"ipc:///tmp/tc{testId}"
        listener = ZmqListener(addr)
        ec_ctx = callable2ctx(callableInstance, addr)
        mp_ctx = platform.get_mp_ctx("executor-aux")
        runner = mp_ctx.Process(target=simple_runner, args=(addr, ec_ctx))
        runner.start()
        output = DatasetId("taskId", "0")

        end = perf_counter_ns() + max_runtime_sec * int(1e9)
        while perf_counter_ns() < end:
            mess = listener.recv_messages()
            if mess == [
                DatasetPublished(origin="testWorker", ds=output, transmit_idx=None)
            ]:
                break
            elif not mess:
                continue
            else:
                raise ValueError(mess)

        runner.join()
        output_buf = shm_client.get(ds2shmid(output))
        output_des = cloudpickle.loads(output_buf.view())
        output_buf.close()
    assert output_des == callableInstance.exp_output


def demo():
    def myfunc(l: int) -> float:
        import numpy as np

        return np.arange(l).sum()

    ci = CallableInstance(
        func=myfunc, kwargs={"l": 4}, args=[], env=["numpy"], exp_output=6
    )
    run_test(ci, "numpyTest1", 2)


if __name__ == "__main__":
    demo()
