# NOTE Copy of cascade.low.func -- dont want to create dependency in either direction. Ideally one func typing util lib

from typing import Any, NoReturn


def assert_never(v: Any) -> NoReturn:
    """For exhaustive enumm checks etc"""
    raise TypeError(v)
