from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from cascade.low.core import JobInstanceRich


@dataclass(frozen=True, eq=True, slots=True)
class JobSpec:
    workers: int
    hosts: int = 1


# receives either the job outputs, or the exception raised by the execution -- allows encoding expected failures
OutputOk = Callable[[Mapping[Any, Any] | Exception], None]


@dataclass(frozen=True, slots=True)
class TestCase:
    __test__ = False  # not a pytest class

    job: JobInstanceRich
    spec: JobSpec
    outputOk: OutputOk


def expect_success(check: Callable[[Mapping[Any, Any]], None]) -> OutputOk:
    """Adapts an outputs-only check: any execution exception is re-raised as unexpected"""

    def wrapped(result: Mapping[Any, Any] | Exception) -> None:
        if isinstance(result, Exception):
            raise result
        check(result)

    return wrapped
