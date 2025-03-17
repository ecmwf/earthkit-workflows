"""
Handles reporting to gateway
"""

import pickle
from cascade.low.core import DatasetId

JobId = str
JobProgress = str

@dataclass
class ControllerReport:
    job_id: JobId
    current_status: JobProgress
    timestamp: int
    results: list[tuple[DatasetId, bytes]]

def deserialize(raw: bytes) -> ControllerReport:
    maybe = pickle.load(raw)
    if isinstance(maybe, ControllerReport):
        return maybe
    else:
        raise TypeError(type(maybe))

def serialize(report: ControllerReport) -> bytes:
    return pickle.dump(report)

def send(report: ControllerReport) -> None:
    # TODO we need to make sure this is reliable, ie, retries and acks from gateway
    raise NotImplementedError
