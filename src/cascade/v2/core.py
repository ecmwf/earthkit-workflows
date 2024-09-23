"""
Core graph data structures -- prescribes most of the API
"""

from base64 import b64decode, b64encode
from typing import Any, Callable, Optional, cast

import cloudpickle
from pydantic import BaseModel, Field

# NOTE it would be tempting to dict[str|int, ...] at places where we deal with kwargs/args, instead of
# double field dict[str] and dict[int]. However, that won't survive serde -- you end up with ints being
# strings


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
    # NOTE missing: cpu, gpu
    memory_mb: int


class Environment(BaseModel):
    # NOTE missing: comm speed etc
    # NOTE hosts are str|int because of Dask, but the int wont survive serde so dont use it
    hosts: dict[str|int, Host]


class Schedule(BaseModel):
    # NOTE hosts are str|int because of Dask, but the int wont survive serde so dont use it
    host_task_queues: dict[str|int, list[str]]
