# ) (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Core graph data structures -- prescribes most of the API"""

import re
from base64 import b64decode, b64encode
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, cast

import cloudpickle
from pydantic import BaseModel, Field
from typing_extensions import Self

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
        description="pip-installable packages, as required by entrypoint/func. Version pins supported"
    )
    # NOTE we could accept eg has_kwargs, has_args, etc... or serialize the whole inspect.signature here?
    input_schema: dict[str, str] = Field(
        description="kv of input kw params and their types (fqn of class). Non-kw params not validated"
    )
    output_schema: list[tuple[str, str]] = Field(
        description="kv of outputs and their types (fqn of class). Assumes listing in func output order"
    )
    needs_gpu: bool = Field(
        False
    )  # NOTE unstable contract, will change. Note we support at most one GPU per task

    @staticmethod
    def func_dec(f: str) -> Callable:
        return cast(Callable, cloudpickle.loads(b64decode(f)))

    @staticmethod
    def func_enc(f: Callable) -> str:
        return b64encode(cloudpickle.dumps(f)).decode("ascii")


TaskId = str


@dataclass(frozen=True)
class DatasetId:
    task: TaskId
    output: str

    def __repr__(self) -> str:
        return f"{self.task}.{self.output}"


class Task2TaskEdge(BaseModel):
    source: DatasetId
    sink_task: TaskId
    sink_input_kw: Optional[str]
    sink_input_ps: Optional[int]


class JobDefinition(BaseModel):
    # NOTE may be redundant altogether as not used rn -- or maybe useful with ProductDefinitions
    definitions: dict[TaskId, TaskDefinition]
    edges: list[Task2TaskEdge]


# Instances
class TaskInstance(BaseModel):
    definition: TaskDefinition
    static_input_kw: dict[str, Any] = Field(
        description="input parameters for the entrypoint. Must be json/msgpack-serializable"
    )
    static_input_ps: dict[str, Any] = Field(
        description="input parameters for the entrypoint. Must be json/msgpack-serializable"
    )


# Type can't be json serialized directly -- use these two functions with `serdes` on JobInstance
def type_dec(t: str) -> Type:
    return cast(Type, cloudpickle.loads(b64decode(t)))


def type_enc(t: Type) -> str:
    return b64encode(cloudpickle.dumps(t)).decode("ascii")


class JobInstance(BaseModel):
    tasks: dict[TaskId, TaskInstance]
    edges: list[Task2TaskEdge]
    serdes: dict[str, tuple[str, str]] = Field(
        {},
        description="for each Type with custom serde, add entry here. The string is fully qualified name of the ser/des functions",
    )
    ext_outputs: list[DatasetId] = Field(
        [], description="ids to externally materialize"
    )

    def outputs_of(self, task_id: TaskId) -> set[DatasetId]:
        return {
            DatasetId(task_id, output)
            for output, _ in self.tasks[task_id].definition.output_schema
        }


HostId = str


@dataclass(frozen=True)
class WorkerId:
    host: HostId
    worker: str

    def __repr__(self) -> str:
        return f"{self.host}.{self.worker}"

    @classmethod
    def from_repr(cls, value: str) -> Self:
        host, worker = value.split(".", 1)
        return cls(host=host, worker=worker)

    def worker_num(self) -> int:
        """Used eg for gpu allocation"""
        # TODO this should actually be precalculated at *Environment* construction, to modulo by gpu count etc
        return int(cast(re.Match[str], re.match("[^0-9]*([0-9]*)", self.worker))[1])


# Execution
class Worker(BaseModel):
    # NOTE we may want to extend cpu/gpu over time with more rich information
    # NOTE keep in sync with executor.msg.Worker
    cpu: int
    gpu: int
    memory_mb: int


class Environment(BaseModel):
    workers: dict[WorkerId, Worker]


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
    tasks: dict[TaskId, TaskExecutionRecord] = Field(
        default_factory=lambda: defaultdict(lambda: no_record_ts)
    )
    datasets_mb: dict[DatasetId, int] = Field(
        default_factory=lambda: defaultdict(lambda: no_record_ds)
    )  # keyed by (task, output)

    # TODO extend this with some approximation/default from TaskInstance only
