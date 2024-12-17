from dataclasses import dataclass
from cascade.controller.core import Action, ActionDatasetPurge, ActionDatasetTransmit, ActionSubmit, TransmitPayload, Event
from cascade.low.core import Environment, WorkerId, DatasetId
from typing import Type
from pydantic import BaseModel
import orjson

class RegisterRequest(BaseModel):
    url: str
    host_id: str
    environment: Environment

class RegisterResponse(BaseModel):
    pass

class Shutdown(BaseModel):
    pass

@dataclass
class DataTransmitObject:
    worker_id: WorkerId
    dataset_id: DatasetId
    data: bytes

class DatasetFetch(BaseModel):
    worker: WorkerId
    dataset: DatasetId

Message = Action | TransmitPayload | Event | RegisterRequest | RegisterResponse | Shutdown | DataTransmitObject | DatasetFetch

b2m: dict[bytes, Type[Message]] = {
    b"\x01": ActionDatasetPurge,
    b"\x02": ActionDatasetTransmit,
    b"\x03": ActionSubmit,
    b"\x04": TransmitPayload,
    b"\x05": Event,
    b"\x06": RegisterRequest,
    b"\x07": RegisterResponse,
    b"\x08": Shutdown,
    b"\x09": DataTransmitObject,
    b"\xa0": DatasetFetch,
}
m2b: dict[Type[Message], bytes] = {v: k for k, v in b2m.items()}

# NOTE we handle serde of DataTransmitObject differently due to being mostly binary data already
# This branching here will disappear once we split message/data on the backbone.interface level

def serialize(message: Message) -> bytes:
    if isinstance(message, DataTransmitObject):
        header = orjson.dumps({"w": message.worker_id, "d": message.dataset_id})
        if b';' in header:
            raise ValueError(header)
        # TODO we dont want to create a bytes here, we want to support message.data being a view
        return m2b[message.__class__] + header + b';' + message.data
    else:
        as_json = message.model_dump()
        return m2b[message.__class__] + orjson.dumps(as_json)

def deserialize(b: bytes) -> Message:
    clazz = b2m[b[:1]]
    if clazz == DataTransmitObject:
        sep = b[1:].index(b';')
        header = orjson.loads(b[1:sep+1])
        return DataTransmitObject(worker_id=header['w'], dataset_id=DatasetId(**header['d']), data=b[sep+2:])
    else:
        as_json = orjson.loads(b[1:])
        return clazz(**as_json)
