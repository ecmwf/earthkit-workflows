"""
Simple facade in front of (any) `controller.executor` implementation. Works in the same
process/thread as the executor itself -- suboptimal especially wrt the data transmit calls
"""

class TransmitPayload(BaseModel):
    # corresponds to ActionDatasetTransmit -- but since it happens across workers, we cant
    # just reuse the original model
    other_url: str
    other_worker: str
    this_worker: str
    datasets: list[DatasetId]


# get
def get_environment():
    raise NotImplementedError

# put
def submit(): # ActionSubmit
    raise NotImplementedError

# post
def transmit(): # TransmitPayload
    raise NotImplementedError

# post
def purge(): # ActionDatasetPurge
    raise NotImplementedError

# get
def wait_some(): # returns: list[Events]
    raise NotImplementedError
