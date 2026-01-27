# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Handles the checkpoint management: storage, retrieval"""

import io
import os
import pathlib

from cascade.executor.msg import DatasetPersistCommand, DatasetRetrieveCommand
from cascade.executor.platform import advise_seqread
from cascade.executor.runner.memory import ds2shmid
from cascade.executor.serde import DefaultSerde
from cascade.low.core import CheckpointSpec, DatasetId
from cascade.low.func import assert_never
from cascade.shm.client import AllocatedBuffer, allocate


def persist_dataset(command: DatasetPersistCommand, buf: AllocatedBuffer) -> None:
    match command.storage_type:
        case "fs":
            root = pathlib.Path(command.persist_params)
            root.mkdir(parents=True, exist_ok=True)
            file = root / command.ds.ser()
            # TODO what about overwrites / concurrent writes? Append uuid?
            file.write_bytes(buf.view())
        case s:
            assert_never(s)

def serialize_params(spec: CheckpointSpec, id_: str) -> str:
    """id_ is either the persist id or retrieve id from the spec"""
    # NOTE we call this every time we store, ideally call this once when building `low.execution_context`
    match spec.storage_type:
        case "fs":
            if not isinstance(spec.storage_params, str):
                raise TypeError(f"expected checkpoint storage params to be str, gotten {spec.storage_params.__class__}")
            root = pathlib.Path(spec.storage_params)
            return str(root / id_)
        case s:
            assert_never(s)

def list_persisted_datasets(spec: CheckpointSpec) -> list[DatasetId]:
    match spec.storage_type:
        case "fs":
            root = pathlib.Path(spec.storage_params)
            files = (x for x in root.iterdir() if x.is_file())
            return [DatasetId.des(file.parts[-1]) for file in files]
        case s:
            assert_never(s)

def retrieve_dataset(command: DatasetRetrieveCommand) -> AllocatedBuffer:
    match command.storage_type:
        case "fs":
            shm_key = ds2shmid(command.ds)
            fpath = pathlib.Path(command.retrieve_params) / command.ds.ser()
            fd = os.open(fpath, os.O_RDONLY)
            try:
                advise_seqread(fd)
                size = os.fstat(fd).st_size
                # TODO dont use default serde, get it via the command
                buf = allocate(shm_key, size, DefaultSerde)
                # once on 3.14+, replace with this
                # os.readinto(fd, buf.view())
                with io.FileIO(fd, closefd=False) as raw_io:
                    raw_io.readinto(buf.view())
            finally:
                os.close(fd)
            return buf
        case s:
            assert_never(s)
