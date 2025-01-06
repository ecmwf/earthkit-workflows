from multiprocessing import Process
import socket
from logging.config import dictConfig

from cascade.low.core import JobInstance, WorkerId
from cascade.executor.msg import BackboneAddress, ExecutorRegistration, ExecutorShutdown
from cascade.executor.comms import Listener, callback
from cascade.executor.executor import Executor
from cascade.executor.config import logging_config

def launch_executor(job_instance: JobInstance, controller_address: BackboneAddress, portBase: int):
    dictConfig(logging_config)
    executor = Executor(job_instance, controller_address, 4, "test_executor", portBase)
    executor.register()
    executor.recv_loop()
    
def test_executor():
    no_job = JobInstance(tasks={}, edges=[])
    c1 = "tcp://localhost:12345"
    e1 = f"tcp://{socket.gethostname()}:12346"
    l = Listener(c1)
    p = Process(target=launch_executor, args=(no_job, c1, 12346))
    p.start()
    try:
        ms = l.recv_messages()
        assert ms == [ExecutorRegistration(host='test_executor', address=e1, workers=[
            WorkerId("test_executor", f"w{i}") for i in range(4)
        ])]

        callback(e1, ExecutorShutdown())

        p.join()
    except Exception as e:
        if p.is_alive():
            p.kill()
        raise
