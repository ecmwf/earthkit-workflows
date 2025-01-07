"""
Here we spin a fullblown executor instance, submit a job to it, and observe that the right zmq messages
get sent out. In essence, we build a pseudo-controller in this test.
"""

from multiprocessing import Process
import socket
from logging.config import dictConfig

from cascade.low.core import JobInstance, WorkerId, TaskDefinition, TaskInstance, Task2TaskEdge, DatasetId
from cascade.executor.msg import BackboneAddress, ExecutorRegistration, ExecutorShutdown, TaskSequence, ExecutorExit, TaskSuccess, DatasetPublished, DatasetTransmitCommand, DatasetTransmitPayload, DatasetPurge
from cascade.executor.comms import Listener, callback
from cascade.executor.executor import Executor
from cascade.executor.config import logging_config
import cascade.executor.serde as serde

def launch_executor(job_instance: JobInstance, controller_address: BackboneAddress, portBase: int):
    dictConfig(logging_config)
    executor = Executor(job_instance, controller_address, 4, "test_executor", portBase)
    executor.register()
    executor.recv_loop()
    
def test_executor():
    # job
    def test_func(x):
        return x+1
    task_definition = TaskDefinition(
        func=TaskDefinition.func_enc(test_func),
        environment=[],
        input_schema={"x": "int"},
        output_schema={"o": "int"},
    )
    source = TaskInstance(
        definition=task_definition,
        static_input_kw={"x": 1},
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
        tasks={'source': source, 'sink': sink},
        edges=[Task2TaskEdge(source=source_o, sink_task="sink", sink_input_kw="x", sink_input_ps=None)],
    )

    # cluster setup
    c1 = "tcp://localhost:12345"
    m1 = f"tcp://{socket.gethostname()}:12346"
    d1 = f"tcp://{socket.gethostname()}:12347"
    l = Listener(c1) # controller
    p = Process(target=launch_executor, args=(job, c1, 12346))

    # run
    p.start()
    try:
        # register
        ms = l.recv_messages()
        assert ms == [
            ExecutorRegistration(
                host='test_executor',
                maddress=m1,
                daddress=d1,
                workers=[
                    WorkerId("test_executor", f"w{i}") for i in range(4)
                ],
            ),
        ]

        # submit graph
        w0 = WorkerId("test_executor", "w0")
        callback(m1, TaskSequence(worker=w0, tasks=["source", "sink"], publish={sink_o}))
        expected = {
            TaskSuccess(worker=w0, ts='source'),
            DatasetPublished(host='test_executor', ds=sink_o),
            TaskSuccess(worker=w0, ts='sink'),
        }
        while expected:
            ms = l.recv_messages()
            for m in ms:
                expected.remove(m)

        # retrieve result
        callback(d1, DatasetTransmitCommand(ds=sink_o, source="test_executor", target="controller", daddress=c1))
        ms = l.recv_messages()
        assert len(ms) == 1 and isinstance(ms[0], DatasetTransmitPayload) and ms[0].ds == DatasetId(task='sink', output='o')
        assert serde.des_output(ms[0].value, 'int') == 3

        # purge, store, run partial and fetch again
        callback(m1, DatasetPurge(ds=sink_o))
        callback(d1, DatasetTransmitPayload(ds=source_o, value=serde.ser_output(10, 'int')))
        ms = l.recv_messages()
        assert ms == [DatasetPublished(host='test_executor', ds=source_o)]
        callback(m1, TaskSequence(worker=w0, tasks=["sink"], publish={sink_o}))
        expected = [
            DatasetPublished(host='test_executor', ds=sink_o),
            TaskSuccess(worker=w0, ts='sink'),
        ]
        while expected:
            ms = l.recv_messages()
            for m in ms:
                assert m == expected[0]
                expected.pop(0)
        callback(d1, DatasetTransmitCommand(ds=sink_o, source="test_executor", target="controller", daddress=c1))
        ms = l.recv_messages()
        assert len(ms) == 1 and isinstance(ms[0], DatasetTransmitPayload) and ms[0].ds == DatasetId(task='sink', output='o')
        assert serde.des_output(ms[0].value, 'int') == 11

        # shutdown
        callback(m1, ExecutorShutdown())
        ms = l.recv_messages()
        assert ms == [ExecutorExit(host='test_executor')]
        p.join()
    except Exception as e:
        if p.is_alive():
            callback(m1, ExecutorShutdown())
            import time
            time.sleep(1)
            p.kill()
        raise
