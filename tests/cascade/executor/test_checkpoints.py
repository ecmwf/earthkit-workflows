import pathlib
import tempfile
from unittest.mock import patch

from cascade.executor.checkpoints import (
    build_persist_command,
    build_retrieve_command,
    list_persisted_datasets,
    persist_dataset,
    retrieve_dataset,
)
from cascade.executor.serde import DefaultSerde
from cascade.low.core import CheckpointSpec, DatasetId
from cascade.low.execution_context import VirtualCheckpointHost
from cascade.shm.client import AllocatedBuffer


def test_rw():
    with tempfile.TemporaryDirectory() as td:
        ds1 = DatasetId(task="1", output="0")
        spec = CheckpointSpec(
            storage_type="fs",
            storage_params=td,
            retrieve_id="subfolder",
            persist_id="subfolder",
            to_persist=[ds1],
        )

        command_persist = build_persist_command(spec, ds1, VirtualCheckpointHost)
        command_retrieve = build_retrieve_command(spec, ds1, VirtualCheckpointHost)
        # we manually create because we dont have an shm server running
        data = AllocatedBuffer("testCascExecChckptRW_1", 64, True, None, DefaultSerde)
        try:
            data.view()[:] = b"a" * 64
            persist_dataset(command_persist, data)

            assert list_persisted_datasets(spec) == [ds1]
            root = pathlib.Path(td) / "subfolder"
            assert root.exists() and root.is_dir()
            assert [e for e in root.iterdir()] == [root / ds1.ser()]

            # we need to patch because we dont have an shm server running
            with patch("cascade.executor.checkpoints.allocate") as mock_allocate:
                allocated = AllocatedBuffer("testCascExecChckptRW_2", 64, True, None, DefaultSerde)
                try:
                    mock_allocate.return_value = allocated
                    retrieved = retrieve_dataset(command_retrieve)
                    assert retrieved.view()[:] == data.view()[:]
                finally:
                    if allocated.shm is not None:
                        allocated.shm.unlink()
                    allocated.close()
        finally:
            if data.shm is not None:
                data.shm.unlink()
            data.close()
