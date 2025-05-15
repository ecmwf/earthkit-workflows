# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Here we spin a fullblown executor instance, submit a job to it, and observe that the right zmq messages
get sent out. In essence, we build a pseudo-controller in this test.
"""

import logging
import socket
from logging.config import dictConfig
from multiprocessing import Process

import numpy as np

import cascade.executor.serde as serde
from cascade.executor.comms import Listener, callback, send_data
from cascade.executor.config import logging_config
from cascade.executor.executor import Executor
from cascade.executor.msg import (
    Ack,
    BackboneAddress,
    DatasetPublished,
    DatasetPurge,
    DatasetTransmitCommand,
    DatasetTransmitPayload,
    DatasetTransmitPayloadHeader,
    ExecutorExit,
    ExecutorRegistration,
    ExecutorShutdown,
    Syn,
    TaskSequence,
    Worker,
)
from cascade.low.core import (
    DatasetId,
    JobInstance,
    Task2TaskEdge,
    TaskDefinition,
    TaskInstance,
    WorkerId,
)

logger = logging.getLogger(__name__)


def launch_executor(
    job_instance: JobInstance, controller_address: BackboneAddress, portBase: int
):
    dictConfig(logging_config)
    executor = Executor(job_instance, controller_address, 4, "test_executor", portBase)
    executor.register()
    executor.recv_loop()


def test_executor():
    # job
    def test_func(x: np.ndarray) -> np.ndarray:
        return x + 1

    task_definition = TaskDefinition(
        func=TaskDefinition.func_enc(test_func),
        environment=[],
        input_schema={"x": "ndarray"},
        output_schema=[("o", "ndarray")],
    )
    source = TaskInstance(
        definition=task_definition,
        static_input_kw={"x": np.array([1.0])},
        static_input_ps={},
    )
    source_o = DatasetId("source", "o")
    sink = TaskInstance(
        definition=task_definition,
        static_input_kw={},
        static_input_ps={},
    )
    sink_o = DatasetId("sink", "o")
    job = JobInstance(
        tasks={"source": source, "sink": sink},
        edges=[
            Task2TaskEdge(
                source=source_o, sink_task="sink", sink_input_kw="x", sink_input_ps=None
            )
        ],
    )

    # cluster setup
    c1 = "tcp://localhost:12545"
    m1 = f"tcp://{socket.gethostname()}:12546"
    d1 = f"tcp://{socket.gethostname()}:12547"
    l = Listener(c1)  # controller
    p = Process(target=launch_executor, args=(job, c1, 12546))

    # run
    p.start()
    try:
        # register
        ms = l.recv_messages(None)
        expected_registration = ExecutorRegistration(
            host="test_executor",
            maddress=m1,
            daddress=d1,
            workers=[
                Worker(
                    worker_id=WorkerId("test_executor", f"w{i}"),
                    cpu=1,
                    gpu=0,
                    memory_mb=1024,
                )
                for i in range(4)
            ],
        )
        assert len(ms) >= 1
        for m in ms:
            # we may receive the registration multiple times due to retries
            assert m == expected_registration

        # submit graph
        w0 = WorkerId("test_executor", "w0")
        callback(
            m1, TaskSequence(worker=w0, tasks=["source", "sink"], publish={sink_o})
        )
        expected = {
            DatasetPublished(origin=w0, ds=sink_o, transmit_idx=None),
        }
        while expected:
            ms = l.recv_messages()
            for m in ms:
                if not isinstance(
                    m, ExecutorRegistration
                ):  # there may be extra due to retries
                    expected.remove(m)

        # retrieve result
        callback(
            d1,
            DatasetTransmitCommand(
                ds=sink_o,
                idx=0,
                source="test_executor",
                target="controller",
                daddress=c1,
            ),
        )
        ms = l.recv_messages()
        assert (
            len(ms) == 1
            and isinstance(ms[0], DatasetTransmitPayload)
            and ms[0].header.ds == DatasetId(task="sink", output="o")
        )
        assert serde.des_output(ms[0].value, "int", ms[0].header.deser_fun)[0] == 3.0

        # purge, store, run partial and fetch again
        callback(m1, DatasetPurge(ds=sink_o))
        value, deser_fun = serde.ser_output(np.array([10.0]), "ndarray")
        payload = DatasetTransmitPayload(
            header=DatasetTransmitPayloadHeader(
                ds=source_o, confirm_idx=1, confirm_address=c1, deser_fun=deser_fun
            ),
            value=value,
        )
        syn = Syn(1, c1)
        send_data(d1, payload, syn)
        expected = {
            Ack(idx=1),
            DatasetPublished(origin="test_executor", ds=source_o, transmit_idx=1),
        }
        while expected:
            ms = l.recv_messages()
            for m in ms:
                logger.debug(f"about to remove received message {m}")
                expected.remove(m)
        callback(m1, TaskSequence(worker=w0, tasks=["sink"], publish={sink_o}))
        expected = [
            DatasetPublished(w0, ds=sink_o, transmit_idx=None),
        ]
        while expected:
            ms = l.recv_messages()
            for m in ms:
                assert m == expected[0]
                expected.pop(0)
        callback(
            d1,
            DatasetTransmitCommand(
                ds=sink_o,
                idx=2,
                source="test_executor",
                target="controller",
                daddress=c1,
            ),
        )
        ms = l.recv_messages()
        assert all(isinstance(m, ExecutorRegistration) for m in ms)
        # NOTE the below ceased to work since we introduced retries. Now recomputation of a result after a purge is not possible
        # assert len(ms) == 1 and isinstance(ms[0], DatasetTransmitPayload) and ms[0].header.ds == DatasetId(task='sink', output='o')
        # assert serde.des_output(ms[0].value, 'ndarray', ms[0].header.deser_fun)[0] == 11.
        # callback(ms[0].header.confirm_address, DatasetTransmitConfirm(idx=ms[0].header.confirm_idx))

        # shutdown
        callback(m1, ExecutorShutdown())
        ms = l.recv_messages()
        assert ms == [ExecutorExit(host="test_executor")]
        p.join()
    except:
        if p.is_alive():
            callback(m1, ExecutorShutdown())
            import time

            time.sleep(1)
            p.kill()
        raise
