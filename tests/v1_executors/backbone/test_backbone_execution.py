"""
For a given graph, naive scheduler and backbone executor, check that things complete
"""

import logging
import time
from cascade.controller.impl import run
from multiprocessing import Process
from cascade.executors.instant import InstantExecutor
from cascade.executors.backbone.local import BackboneLocalExecutor
from cascade.executors.backbone.executor import BackboneExecutor
from cascade.executors.backbone.zmq import ZmqBackbone
from cascade.graph import Node
from cascade.low.builders import JobBuilder, TaskBuilder
from cascade.low.core import Environment, JobExecutionRecord, TaskExecutionRecord, JobInstance, Worker, WorkerId
from cascade.scheduler.graph import precompute
import cascade.shm.server as shm_server
import cascade.shm.api as shm_api
import cascade.shm.client as shm_client
from cascade.controller.executor import Executor
from cascade.executors.simulator import SimulatingExecutor, placeholder_execution_record
from cascade.low.views import param_source

logger = logging.getLogger(__name__)

def launch_executor(port_start: int, idx: int, job: JobInstance, kind: str) -> None:
    from forecastbox.utils import logging_config as fiab_logging
    logging.config.dictConfig(fiab_logging)
    controller_url = f"tcp://localhost:{port_start}"
    executor: Executor
    if kind == "fiab":
        shm_shift = 4
        if shm_shift <= idx:
            raise ValueError
        from forecastbox.executor.executor import SingleHostExecutor, Config as ExecutorConfig
        from forecastbox.utils import logging_config as fiab_logging
        gb4 = 4 * (1024**3)
        shm_port = port_start + idx + shm_shift
        shm_api.publish_client_port(shm_port)
        shm_pref = f"s{idx}_"
        shm = Process(target=shm_server.entrypoint, args=(shm_port, gb4, fiab_logging, shm_pref))
        shm.start()
        shm_client.ensure()
        executor = SingleHostExecutor(ExecutorConfig(1, 1024, f"h{idx}"), job)
    elif kind == "instant":
        executor = InstantExecutor(workers=1, job=job, host_id=f"h{idx}")
    elif kind == "simulating":
        env = Environment(
            workers={WorkerId(f"h{idx}","w0"): Worker(cpu=1, gpu=0, memory_mb=1024)},
            colocations=[[f"h{idx}:w0"]],
        )
        task_inputs = {
            task_id: set(task_param_source.values())
            for task_id, task_param_source in param_source(job.edges).items()
        }
        record = placeholder_execution_record(job)
        executor = SimulatingExecutor(env, task_inputs, record)
    else:
        raise NotImplementedError(kind)

    self_url = f"tcp://localhost:{port_start+idx}"
    backbone = ZmqBackbone(self_url, controller_url=controller_url, environment=executor.get_environment(), host_id=f"h{idx}")
    adapter = BackboneLocalExecutor(executor, backbone)
    adapter.recv_loop()
    if kind == "fiab":
        shm_client.shutdown()
        shm.terminate()

def launch_controller(port_start: int, job: JobInstance, workers: int) -> None:
    from forecastbox.utils import logging_config as fiab_logging
    logging.config.dictConfig(fiab_logging)
    controller_url = f"tcp://localhost:{port_start}"
    backbone = ZmqBackbone(controller_url, expected_workers=workers)
    executor = BackboneExecutor(backbone)
    preschedule = precompute(job)
    run(job, executor, preschedule)

def launch_and_run(port_start: int, job: JobInstance, kind: str, workers: int) -> None:
    threads = []
    ctrl_thread = Process(target=launch_controller, args=(port_start, job, workers))
    ctrl_thread.start()
    for idx in range(workers):
        threads.append(Process(target=launch_executor, args=(port_start, idx+1, job, kind)))
        threads[idx].start()
    ctrl_thread.join()
    logger.debug(f"ctrl process joined")
    for t in threads:
        t.join()
        logger.debug(f"worker process joined")

def test_simple() -> None:
    def simple_func(a: int, b: int) -> int:
        return a + b

    # 2-node graph
    task1 = TaskBuilder.from_callable(simple_func).with_values(a=1, b=2)
    task2 = TaskBuilder.from_callable(simple_func).with_values(a=1)
    job = (
        JobBuilder()
        .with_node("task1", task1)
        .with_node("task2", task2)
        .with_edge("task1", "task2", "b")
        .build()
        .get_or_raise()
    )

    port_start = 5565
    logger.debug(f"running test with instant executor")
    # launch_and_run(port_start, job, "instant", 1)
    logger.debug(f"running test with fiab executor")
    launch_and_run(port_start + 10, job, "fiab", 1)
    logger.debug(f"running test with simulating executor")
    launch_and_run(port_start + 20, job, "simulating", 1)

def test_transmit() -> None:
    def data_gen(a: int) -> int:
        return a*2

    def data_cons(a: int, b: int) -> int:
        # this should force transmit
        time.sleep(0.5)
        return a+b

    task1 = TaskBuilder.from_callable(data_gen).with_values(a=1)
    task2a = TaskBuilder.from_callable(data_cons).with_values(b=1)
    task2b = TaskBuilder.from_callable(data_cons).with_values(b=2)

    job = (
        JobBuilder()
        .with_node("task1", task1)
        .with_node("task2a", task2a)
        .with_node("task2b", task2b)
        .with_edge("task1", "task2a", "a")
        .with_edge("task1", "task2b", "a")
        .build()
        .get_or_raise()
    )
    port_start = 5575
    # TODO the problem here is that the instant executor finishes too fast, and not all
    # messages from the client get accepted by the controller. So even though the run is
    # ok, the process remains hanging at the end
    launch_and_run(port_start, job, "instant", 2)
    launch_and_run(port_start + 10, job, "fiab", 2)
    launch_and_run(port_start + 20, job, "simulating", 2)

def test_failure() -> None:
    """Tests that the cluster correctly exists when a task crashes, as opposed to hanging
    infinitely"""
    def i_may_crash(a: int) -> int:
        if a == 0:
            raise ValueError("zero crash")
        else:
            return 1

    def i_wait(a: int) -> int:
        import time
        time.sleep(0.1)
        return 1

    task1 = TaskBuilder.from_callable(i_may_crash).with_values(a=0)
    job1 = (
        JobBuilder()
        .with_node("task1", task1)
        .build()
        .get_or_raise()
    )

    port_start = 5575
    launch_and_run(port_start, job1, "fiab", 1)

    task2 = TaskBuilder.from_callable(i_wait).with_values(a=0)
    job2 = (
        JobBuilder()
        .with_node("task1", task1)
        .with_node("task2", task2)
        .build()
        .get_or_raise()
    )
    launch_and_run(port_start + 10, job2, "fiab", 2)
