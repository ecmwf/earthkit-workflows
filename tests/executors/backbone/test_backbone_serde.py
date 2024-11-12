from cascade.low.core import DatasetId, Environment
from cascade.controller.core import ActionSubmit, ActionDatasetPurge, Event, DatasetStatus, TaskStatus, ActionDatasetTransmit
from cascade.executors.multihost.worker_server import TransmitPayload
import orjson
from typing import TypeVar
from pydantic import BaseModel
from cascade.executors.backbone.serde import Message, serialize, deserialize, RegisterRequest, RegisterResponse, Shutdown, DataTransmitObject

M = TypeVar("M", bound=Message)
def there_and_back_again(m: M) -> None:
    s = serialize(m)
    r = deserialize(s)
    assert m == r

def test_serde():
    submit = ActionSubmit(at="worker1", tasks=["t1", "t2"], outputs={DatasetId("t1", "o1"), DatasetId("t2", "o2")})
    there_and_back_again(submit)

    purge = ActionDatasetPurge(at={"worker1", "worker2"}, ds={DatasetId("t1", "o1"), DatasetId("t2", "o2")})
    there_and_back_again(purge)

    event = Event(
        at="worker1",
        ds_trans=[(DatasetId("t1", "o1"), DatasetStatus.available)],
        ts_trans=[("task1", TaskStatus.succeeded)],
    )
    there_and_back_again(event)
    
    transmit_loc = ActionDatasetTransmit(fr=["worker1"], to=["worker2"], ds=[DatasetId("t1", "o1")])
    there_and_back_again(transmit_loc)

    transmit_rem = TransmitPayload(
        other_url="http://localhost:123",
        other_worker="worker1",
        this_worker="worker2",
        datasets=[DatasetId("t1", "o1"), DatasetId("t2", "o2")],
        tracing_ctx_host="host1",
    )
    there_and_back_again(transmit_rem)

    register_req = RegisterRequest(
        url="tcp://localhost:123",
        host_id="h1",
        environment=Environment(
            workers={},
        ),
    )
    there_and_back_again(register_req)
    there_and_back_again(RegisterResponse())
    there_and_back_again(Shutdown())
    there_and_back_again(DataTransmitObject(
        worker_id="h0:w1",
        dataset_id=DatasetId("t1", "o1"),
        data=b";;123abc;;@@",
    ))
