from abc import abstractmethod
from pydantic import BaseModel
from typing import (
    Type,
    Any,
    Callable,
    Generic,
    Iterable,
    NoReturn,
    Optional,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

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
