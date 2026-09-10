# (C) Copyright 2026- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    def trivial(array):
        return array

    @staticmethod
    @abstractmethod
    def mean(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def std(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def max(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def min(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def sum(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def prod(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def var(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def stack(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def concat(*array, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def add(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def subtract(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def multiply(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def divide(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def pow(arr1, arr2, *, backend_kwargs: Optional[dict] = None):
        pass

    @staticmethod
    @abstractmethod
    def take(array, indices, dim: Optional[str | int] = None, *, backend_kwargs: Optional[dict] = None):
        pass
