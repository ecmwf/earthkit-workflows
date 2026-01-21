# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Handles the checkpoint management: storage, retrieval"""

import pathlib

from cascade.executor.msg import DatasetPersistCommand
from cascade.low.core import CheckpointSpec
from cascade.low.func import assert_never
from cascade.shm.client import AllocatedBuffer


def persist_dataset(command: DatasetPersistCommand, buf: AllocatedBuffer) -> None:
    match command.storage_type:
        case "fs":
            root = pathlib.Path(command.persist_params)
            root.mkdir(parents=True, exist_ok=True)
            file = root / repr(command.ds)
            # TODO what about overwrites / concurrent writes? Append uuid?
            file.write_bytes(buf.view())
        case s:
            assert_never(s)

def serialize_persist_params(spec: CheckpointSpec) -> str:
    # NOTE we call this every time we store, ideally call this once when building `low.execution_context`
    match spec.storage_type:
        case "fs":
            if not isinstance(spec.storage_params, str):
                raise TypeError(f"expected checkpoint storage params to be str, gotten {spec.storage_params.__class__}")
            if spec.persist_id is None:
                raise TypeError(f"serialize_persist_params called, but persist_id is None")
            root = pathlib.Path(spec.storage_params)
            return str(root / spec.persist_id)
        case s:
            assert_never(s)

