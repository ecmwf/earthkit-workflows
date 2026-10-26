# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Basic mocked tests of shm failure propagation"""

from typing import cast
from unittest.mock import MagicMock

import pytest

from cascade.executor.executor import Executor
from cascade.executor.runner.memory import shm_guard
from cascade.low.core import DatasetId, TaskId
from cascade.low.exceptions import CascadeInfrastructureError, CascadeInternalError, CascadeUserError, des, ser
from cascade.shm.func import ShmInfrastructureError, ShmInternalError

ds = DatasetId(cast(TaskId, "t1"), "out")


def test_shm_guard_converts_with_context() -> None:
    with pytest.raises(CascadeInfrastructureError, match="get of dataset") as ei:
        with shm_guard(ds, "get"):
            raise ShmInfrastructureError("insufficient space")
    assert "t1" in str(ei.value)
    assert isinstance(ei.value.parent, ShmInfrastructureError)

    with pytest.raises(CascadeInternalError):
        with shm_guard(ds, "get"):
            raise ShmInternalError("bad")

    with pytest.raises(CascadeInfrastructureError):
        with shm_guard(ds, "publish"):
            raise ConnectionRefusedError()


def test_shm_guard_passes_through_cascade_errors() -> None:
    with pytest.raises(CascadeUserError):
        with shm_guard(ds, "get"):
            raise CascadeUserError("x")


def test_executor_shm_status() -> None:
    ex = cast(Executor, object.__new__(Executor))
    ex.shm_process = MagicMock()  # ty: ignore[invalid-assignment]
    ex.shm_process.is_alive.return_value = True
    assert ex._shm_status() == {}
    ex.shm_process.is_alive.return_value = False
    ex.shm_process.pid = 42
    ex.shm_process.exitcode = -9
    status = ex._shm_status()
    assert "exitcode=-9" in status["shm"]
    # extends the original failure instead of replacing it
    extended = des(ser(des(ser(CascadeUserError("orig"))), status))
    assert isinstance(extended, CascadeUserError)
    assert extended.description == "orig"
    assert extended.context["shm"] == status["shm"]
