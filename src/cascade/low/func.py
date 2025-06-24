# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Things."""

import importlib
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Container,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    NoReturn,
    Optional,
    Protocol,
    Type,
    TypeVar,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel
from typing_extensions import Self

T = TypeVar("T")


def maybe_head(v: Iterable[T]) -> Optional[T]:
    try:
        return next(iter(v))
    except StopIteration:
        return None


def assert_never(v: Any) -> NoReturn:
    """For exhaustive enumm checks etc"""
    raise TypeError(v)


@runtime_checkable
class Semigroup(Protocol):
    """Basically 'has plus'"""

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        pass


U = TypeVar("U")
E = TypeVar("E", bound=Semigroup)


class Either(Generic[T, E]):
    """Mostly for lazy gathering of errors during validation. Looks fancier than actually is"""

    def __init__(self, t: Optional[T] = None, e: Optional[E] = None):
        self.t = t
        self.e = e

    @classmethod
    def ok(cls, t: T) -> Self:
        return cls(t=t)

    @classmethod
    def error(cls, e: E) -> Self:
        return cls(e=e)

    def get_or_raise(self, raiser: Optional[Callable[[E], BaseException]] = None) -> T:
        if self.e:
            if not raiser:
                raise ValueError(self.e)
            else:
                raise raiser(self.e)
        else:
            return cast(T, self.t)

    def chain(self, f: Callable[[T], "Either[U, E]"]) -> "Either[U, E]":
        if self.e:
            return self.error(self.e)  # type: ignore # needs higher python and more magic
        else:
            return f(cast(T, self.t))

    def append(self, other: Optional[E]) -> Self:
        if other:
            if not self.e:
                return self.error(other)
            else:
                return self.error(self.e + other)
        else:
            return self


def ensure(l: list, i: int) -> None:
    """Ensures list l has at least i elements, for a safe l[i] = ..."""
    if (k := (i + 1 - len(l))) > 0:
        l.extend([None] * k)


@runtime_checkable
class Monoid(Protocol):

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(cls) -> Self:
        raise NotImplementedError


TMonoid = TypeVar("TMonoid", bound=Monoid)


def msum(i: Iterable[TMonoid], t: Type[TMonoid]) -> TMonoid:
    return sum(i, start=t.empty())


B = TypeVar("B", bound=BaseModel)


def pyd_replace(model: B, **kwargs) -> B:
    """Like dataclasses.replace but for pydantic"""
    return model.model_copy(update=kwargs)


def assert_iter_empty(i: Iterator) -> bool:
    try:
        _ = next(i)
    except StopIteration:
        return True
    else:
        return False


def next_uuid(s: Container[T], g: Callable[[], T]) -> T:
    while True:
        n = g()
        if n not in s:
            return n


def resolve_callable(s: str) -> Callable:
    """For s = `a.b.func`, imports `a.b` and retrieves `func` Callable object"""
    if "." not in s:  # this branch is for builtins
        return eval(s)
    else:
        module_name, function_name = s.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return module.__dict__[function_name]


def pydantic_recursive_collect(
    base: BaseModel | Iterable, attr: str, prefix: str = "."
) -> list[tuple[str, Any]]:
    """Recurse into base, visiting each sub-model, list, dictionary, etc, invoking `attr` if present
    and collecting results. Assumes the `attr` has signature `Callable[[self], list[T]]`. The collected
    results are (source, item), where item is returned by `attr` and source is the dot-separated path
    of the issuer -- eg if base has BaseModel field `x` whose `attr` yields [1, 2], then this returns
    [(.x, 1), (.x, 2)]
    """

    # NOTE a bit ugly, instead of attr it would be better to accept a signature/protocol type

    results: list[str] = []
    if hasattr(base, attr):
        results.extend((prefix, e) for e in getattr(base, attr)())
    generator: Iterable[tuple[Any, Any]]
    if isinstance(base, BaseModel):
        generator = base
        formatter = lambda k: f"{k}."
    elif isinstance(base, Mapping):
        generator = base.items()
        formatter = lambda k: f"{k}."
    elif isinstance(base, Iterable):
        generator = enumerate(base)
        formatter = lambda k: f"[{k}]."
    else:
        return results
    for k, v in generator:
        # we exclude the string as a circuit-breaker optimisation
        if isinstance(v, str):
            continue
        if isinstance(v, BaseModel) or isinstance(v, Iterable):
            results.extend(pydantic_recursive_collect(v, attr, prefix + formatter(k)))
    return results
