from cascade.low.core import DatasetId
from cascade.controller.core import ActionSubmit, ActionDatasetPurge, Event, DatasetStatus, TaskStatus, ActionDatasetTransmit
from cascade.executors.multihost.worker_server import TransmitPayload
import orjson
from typing import TypeVar
from pydantic import BaseModel

B = TypeVar("B", bound=BaseModel)
def there_and_back_again(b: B) -> B:
    dump = b.model_dump()
    raw = orjson.dumps(dump)
    proc = orjson.loads(raw)
    model = b.__class__(**proc)
    assert b == model

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
