"""
Tests running a Callable in the same process
"""

from cascade.executor.msg import TaskSequence, ExecutionContext, TaskSuccess, DatasetPublished
from cascade.low.core import WorkerId, TaskDefinition, TaskInstance, DatasetId
import cascade.executor.serde as serde
import cascade.shm.client as shm_cli
import cascade.executor.runner as runner
import pytest

def test_runner(monkeypatch):
    worker = WorkerId("h0", "w0")

    # monkeypatching
    monkeypatch.setattr(shm_cli, "is_unregister", False)
    test_address = "zmq:test"
    msgs = []
    def verify_msg(address, msg):
        assert address == test_address
        msgs.append(msg)
    monkeypatch.setattr(runner, "callback", verify_msg)

    def allocate(key: str, l: int, timeout_sec: float = 60.0) -> shm_cli.AllocatedBuffer:
        return shm_cli.AllocatedBuffer(shmid=f"test_{key}", l=l, create=True, close_callback=lambda : None)
    monkeypatch.setattr(shm_cli, "allocate", allocate)
    def get(key: str, timeout_sec: float = 60.0) -> shm_cli.AllocatedBuffer:
        return shm_cli.AllocatedBuffer(shmid=f"test_{key}", l=1024, create=False, close_callback=lambda : None)
    monkeypatch.setattr(shm_cli, "get", get)

    # test 1: no tasks
    emptyTs = TaskSequence(
        worker=worker,
        tasks=[],
        publish=set(),
    )
    emptyEc = ExecutionContext(
        tasks={},
        param_source={},
        callback=test_address,
    )

    runner.entrypoint(emptyTs, emptyEc)
    assert msgs == []

    def test_func(x):
        return x+1

    # test 2: one task with a single static input and a published output
    task_definition = TaskDefinition(
        func=TaskDefinition.func_enc(test_func),
        environment=[],
        input_schema={"x": "int"},
        output_schema={"o": "int"},
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
    oneTaskEc = ExecutionContext(
        tasks = {"t2": t2},
        param_source = {"t2": {}},
        callback=test_address,
    )

    runner.entrypoint(oneTaskTs, oneTaskEc)
    assert msgs == [
        DatasetPublished(host=worker.host, ds=t2ds, transmit_idx=None),
        TaskSuccess(worker=worker, ts='t2')
    ]
    msgs = []
    so = get(runner.ds2shmid(t2ds))
    assert serde.des_output(so.view(), 'int') == 2
    so.close()

    # test 3: two task pipeline, utilizing previous pipeline's output
    t3a = TaskInstance(
        definition=task_definition,
        static_input_kw={},
        static_input_ps={},
    )
    t3b = TaskInstance(
        definition=task_definition,
        static_input_kw={},
        static_input_ps={},
    )
    t3ds = DatasetId("t3b", "o")
    twoTaskTs = TaskSequence(
        worker=worker,
        tasks=["t3a", "t3b"],
        publish={t3ds},
    )
    twoTaskEc = ExecutionContext(
        tasks = {"t3a": t3a, "t3b": t3b},
        param_source = {"t3a": {"x": (t2ds, "int")}, "t3b": {"x": (DatasetId("t3a", "o"), 'int')}},
        callback=test_address,
    )

    runner.entrypoint(twoTaskTs, twoTaskEc)
    assert msgs == [
        TaskSuccess(worker=worker, ts='t3a'),
        DatasetPublished(host=worker.host, ds=t3ds, transmit_idx=None),
        TaskSuccess(worker=worker, ts='t3b'),
    ]
    so = get(runner.ds2shmid(t3ds))
    assert serde.des_output(so.view(), 'int') == 4
    so.close()
