"""
Core graph data structures -- prescribes most of the API
"""

from typing import Any, Callable, cast
import cloudpickle
from base64 import b64decode, b64encode

from pydantic import BaseModel, Field


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
    input_schema: dict[str, str] = Field(
        description="kv of input params and their types (fqn of class)"
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
    sink_input: str


class JobDefinition(BaseModel):
    # NOTE may be redundant altogether as not used rn -- or maybe useful with ProductDefinitions
    definitions: dict[str, TaskDefinition]
    edges: list[Task2TaskEdge]


# Instances
class TaskInstance(BaseModel):
    definition: TaskDefinition
    static_input: dict[str, Any] = Field(
        description="input parameters for the entrypoint. Must be json/msgpack-serializable"
    )


class JobInstance(BaseModel):
    tasks: dict[str, TaskInstance]
    edges: list[Task2TaskEdge]


# Execution
class Host(BaseModel):
    # NOTE missing: cpu, gpu
    memory_mb: int


class Environment(BaseModel):
    # NOTE missing: comm speed
    hosts: dict[str, Host]


class Schedule(BaseModel):
    host_task_queues: dict[str, list[str]]
