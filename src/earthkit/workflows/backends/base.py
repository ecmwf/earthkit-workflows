# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from typing import Any, Optional


class Backend(ABC):
    @classmethod
    def trivial(cls, array: Any) -> Any:
        return array

    @classmethod
    @abstractmethod
    def mean(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def std(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def max(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def min(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def sum(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def prod(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def var(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def stack(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concat(cls, *array: Any, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def add(cls, arr1: Any, arr2: Any, *, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def subtract(cls, arr1: Any, arr2: Any, *, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def multiply(cls, arr1: Any, arr2: Any, *, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def divide(cls, arr1: Any, arr2: Any, *, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def pow(cls, arr1: Any, arr2: Any, *, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def take(cls, array: Any, indices: Any, dim: Optional[str | int] = None, *, backend_kwargs: Optional[dict] = None) -> Any:
        raise NotImplementedError
