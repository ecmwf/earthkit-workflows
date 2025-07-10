# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests running a Callable in the same process"""

from multiprocessing.shared_memory import SharedMemory

import cascade.executor.runner.entrypoint as entrypoint
import cascade.executor.runner.memory as memory
import cascade.executor.serde as serde
import cascade.shm.client as shm_cli
from cascade.executor.msg import DatasetPublished, TaskSequence
from cascade.executor.runner.packages import PackagesEnv
from cascade.low.core import (
    DatasetId,
    JobInstance,
    Task2TaskEdge,
    TaskDefinition,
    TaskInstance,
    WorkerId,
)
from cascade.low.views import param_source


def test_runner(monkeypatch):
    worker = WorkerId("h0", "w0")

    # monkeypatching
    monkeypatch.setattr(shm_cli, "is_unregister", False)
    test_address = "zmq:test"
    msgs = []

    def verify_msg(address, msg):
        assert address == test_address
        msgs.append(msg)

    monkeypatch.setattr(memory, "callback", verify_msg)
    monkeypatch.setattr(entrypoint, "callback", verify_msg)

    def allocate(
        key: str, l: int, timeout_sec: float = 60.0
    ) -> shm_cli.AllocatedBuffer:
        shmid = f"test_{key}"
        _allocate = lambda: shm_cli.AllocatedBuffer(
            shmid=shmid,
            l=l,
            create=True,
            close_callback=lambda: None,
            deser_fun="cloudpickle.loads",
        )
        try:
            return _allocate()
        except FileExistsError:
            # for some reason, the is_unregister doesnt seem to cover all cases
            shm = SharedMemory(shmid, create=False)
            shm.unlink()
            shm.close()
            return _allocate()

    monkeypatch.setattr(shm_cli, "allocate", allocate)

    def get(key: str, timeout_sec: float = 60.0) -> shm_cli.AllocatedBuffer:
        return shm_cli.AllocatedBuffer(
            shmid=f"test_{key}",
            l=1024,
            create=False,
            close_callback=lambda: None,
            deser_fun="cloudpickle.loads",
        )

    monkeypatch.setattr(shm_cli, "get", get)

    # test 1: no tasks
    emptyTs = TaskSequence(
        worker=worker,
        tasks=[],
        publish=set(),
    )
    emptyRc = entrypoint.RunnerContext(
        workerId=worker,
        callback=test_address,
        job=JobInstance(tasks={}, edges=[]),
        param_source={},
        log_base=None,
    )

    with memory.Memory(test_address, worker) as memoryInstance, PackagesEnv() as pckg:
        entrypoint.execute_sequence(emptyTs, memoryInstance, pckg, emptyRc)
    assert msgs == []

    def test_func(x):
        return x + 1

    # test 2: one task with a single static input and a published output
    task_definition = TaskDefinition(
        func=TaskDefinition.func_enc(test_func),
        environment=[],
        input_schema={"x": "int"},
        output_schema=[("o", "int")],
    )
    t2 = TaskInstance(
        definition=task_definition,
        static_input_kw={"x": 1},
        static_input_ps={},
    )
    t2ds = DatasetId("t2", "o")
    oneTaskTs = TaskSequence(
        worker=worker,
        tasks=["t2"],
        publish={t2ds},
    )
    oneTaskJob = JobInstance(tasks={"t2": t2}, edges=[])
    oneTaskRc = entrypoint.RunnerContext(
        workerId=worker,
        callback=test_address,
        job=oneTaskJob,
        param_source=param_source(oneTaskJob.edges),
        log_base=None,
    )

    with memory.Memory(test_address, worker) as memoryInstance, PackagesEnv() as pckg:
        entrypoint.execute_sequence(oneTaskTs, memoryInstance, pckg, oneTaskRc)
    assert msgs == [DatasetPublished(origin=worker, ds=t2ds, transmit_idx=None)]
    msgs = []
    so = get(memory.ds2shmid(t2ds))
    assert serde.des_output(so.view(), "int", so.deser_fun) == 2
    so.close()

    # test 3: two task pipeline
    t3a = TaskInstance(
        definition=task_definition,
        static_input_kw={"x": 2},
        static_input_ps={},
    )
    t3b = TaskInstance(
        definition=task_definition,
        static_input_kw={},
        static_input_ps={},
    )
    t3i = DatasetId("t3a", "o")
    t3o = DatasetId("t3b", "o")
    twoTaskTs = TaskSequence(
        worker=worker,
        tasks=["t3a", "t3b"],
        publish={t3o},
    )
    twoTaskJob = JobInstance(
        tasks={"t3a": t3a, "t3b": t3b},
        edges=[
            Task2TaskEdge(
                source=t3i, sink_task="t3b", sink_input_kw="x", sink_input_ps=None
            )
        ],
    )
    twoTaskRc = entrypoint.RunnerContext(
        workerId=worker,
        callback=test_address,
        job=twoTaskJob,
        param_source=param_source(twoTaskJob.edges),
        log_base=None,
    )

    with memory.Memory(test_address, worker) as memoryInstance, PackagesEnv() as pckg:
        entrypoint.execute_sequence(twoTaskTs, memoryInstance, pckg, twoTaskRc)
    assert msgs == [
        DatasetPublished(origin=worker, ds=t3o, transmit_idx=None),
    ]
    msgs = []
    so = get(memory.ds2shmid(t3o))
    assert serde.des_output(so.view(), "int", so.deser_fun) == 4
    so.close()

    # test 4: generator
    N = 4

    def gen_func():
        for i in range(N):
            yield i

    gen_definition = TaskDefinition(
        func=TaskDefinition.func_enc(gen_func),
        environment=[],
        input_schema={},
        output_schema=[(f"{i}", "int") for i in range(N)],
    )
    t4g = TaskInstance(
        definition=gen_definition,
        static_input_kw={},
        static_input_ps={},
    )
    t4gOutputs = [DatasetId("t4g", k) for k, _ in gen_definition.output_schema]
    t4c = TaskInstance(
        definition=task_definition,
        static_input_kw={},
        static_input_ps={},
    )
    t4pOutputs = [DatasetId(f"t4c{i}", "o") for i in range(N)]
    t4TaskTs = TaskSequence(
        worker=worker,
        tasks=["t4g"] + [f"t4c{i}" for i in range(N)],
        publish=set(t4pOutputs),
    )
    t4Job = JobInstance(
        tasks={**{"t4g": t4g}, **{f"t4c{i}": t4c for i in range(N)}},
        edges=[
            Task2TaskEdge(
                source=t4gOutputs[i],
                sink_task=f"t4c{i}",
                sink_input_kw="x",
                sink_input_ps=None,
            )
            for i in range(N)
        ],
    )
    t4Rc = entrypoint.RunnerContext(
        workerId=worker,
        callback=test_address,
        job=t4Job,
        param_source=param_source(t4Job.edges),
        log_base=None,
    )

    with memory.Memory(test_address, worker) as memoryInstance, PackagesEnv() as pckg:
        entrypoint.execute_sequence(t4TaskTs, memoryInstance, pckg, t4Rc)
    assert msgs == [
        DatasetPublished(origin=worker, ds=t4pOutputs[0], transmit_idx=None),
        DatasetPublished(origin=worker, ds=t4pOutputs[1], transmit_idx=None),
        DatasetPublished(origin=worker, ds=t4pOutputs[2], transmit_idx=None),
        DatasetPublished(origin=worker, ds=t4pOutputs[3], transmit_idx=None),
    ]
    for i, o in enumerate(t4pOutputs):
        so = get(memory.ds2shmid(o))
        assert serde.des_output(so.view(), "int", so.deser_fun) == i + 1
        so.close()
