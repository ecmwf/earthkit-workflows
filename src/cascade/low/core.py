"""
Core graph data structures -- prescribes most of the API
"""

from base64 import b64decode, b64encode
from collections import defaultdict
from typing import Any, Callable, Optional, cast

import cloudpickle
from pydantic import BaseModel, Field

# NOTE it would be tempting to dict[str|int, ...] at places where we deal with kwargs/args, instead of
# double field dict[str] and dict[int]. However, that won't survive serde -- you end up with ints being
# strings

# NOTE We want *every* task to have an output, to simplify reasoning wrt completion.
# Thus, if there are no genuine outputs, we insert a `placeholder: str` output
# and expect every executor to generate some value like "ok" in such case
NO_OUTPUT_PLACEHOLDER = "__NO_OUTPUT__"

# Definitions
class TaskDefinition(BaseModel):
    entrypoint: str = Field(
        "",
        description="fqn of a Callable, eg mymod.submod.function. Ignored if `func` given",
    )
    func: str | None = Field(
        None,
        description="a cloud-pickled callable. Prefered over `entrypoint` if given",
    )
    environment: list[str] = Field(
        description="pip-installable packages, should contain entrypoint and all deps it requires"
    )
    # NOTE we could accept eg has_kwargs, has_args, etc... or serialize the whole inspect.signature here?
    input_schema: dict[str, str] = Field(
        description="kv of input kw params and their types (fqn of class). Non-kw params not validated"
    )
    output_schema: dict[str, str] = Field(
        description="kv of outputs and their types (fqn of class)"
    )

    @staticmethod
    def func_dec(f: str) -> Callable:
        return cast(Callable, cloudpickle.loads(b64decode(f)))

    @staticmethod
    def func_enc(f: Callable) -> str:
        return b64encode(cloudpickle.dumps(f)).decode("ascii")


class Task2TaskEdge(BaseModel):
    source_task: str
    source_output: str
    sink_task: str
    sink_input_kw: Optional[str]
    sink_input_ps: Optional[int]


class JobDefinition(BaseModel):
    # NOTE may be redundant altogether as not used rn -- or maybe useful with ProductDefinitions
    definitions: dict[str, TaskDefinition]
    edges: list[Task2TaskEdge]


# Instances
class TaskInstance(BaseModel):
    definition: TaskDefinition
    static_input_kw: dict[str, Any] = Field(
        description="input parameters for the entrypoint. Must be json/msgpack-serializable"
    )
    static_input_ps: dict[int, Any] = Field(
        description="input parameters for the entrypoint. Must be json/msgpack-serializable"
    )


class JobInstance(BaseModel):
    tasks: dict[str, TaskInstance]
    edges: list[Task2TaskEdge]


# Execution
class Host(BaseModel):
    # NOTE we may want to extend cpu/gpu over time with more rich information
    cpu: int
    gpu: int
    memory_mb: int


class Environment(BaseModel):
    # NOTE missing: comm speed etc
    hosts: dict[str, Host]


class TaskExecutionRecord(BaseModel):
    # NOTE rather crude -- we may want to granularize cpuseconds
    cpuseconds: int = Field(
        description="as measured from process start to process end, assuming full cpu util"
    )
    memory_mb: int = Field(
        description="observed rss peak held by the process minus sizes of shared memory inputs"
    )


# possibly made configurable, overridable -- quite job dependent
no_record_ts = TaskExecutionRecord(cpuseconds=1, memory_mb=1)
no_record_ds = 1


class JobExecutionRecord(BaseModel):
    tasks: dict[str, TaskExecutionRecord] = Field(
        default_factory=lambda: defaultdict(lambda: no_record_ts)
    )
    datasets_mb: dict[tuple[str, str], int] = Field(
        default_factory=lambda: defaultdict(lambda: no_record_ds)
    )  # keyed by (task, output)

    # TODO extend this with some approximation/default from TaskInstance only


class Schedule(BaseModel):
    host_task_queues: dict[str, list[list[str]]] # element of task queue is a fused subgraph == list of tasks
    unallocated: set[str] = Field(default_factory=set)

    @classmethod
    def empty(cls):
        return cls(host_task_queues=defaultdict(list))
