import fcntl
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def lock_resource(request, tmp_path_factory):
    """
    Parametrized lock fixture using Unix fcntl.
    Usage: @pytest.mark.concurrency_filelock("my_resource_name")
    """
    # 1. Determine the lock name from the marker, default to 'global'
    marker = request.node.get_closest_marker("concurrency_filelock")
    if not marker:
        yield
        return
    resource_name = marker.args[0]

    # 2. Ensure all xdist workers point to the same base temp directory
    # tmp_path_factory.getbasetemp() is unique per worker,
    # but its parent is shared across the session.
    shared_dir = tmp_path_factory.getbasetemp().parent
    lock_file = shared_dir / f"{resource_name}.lock"

    # 3. Perform the system-level lock
    with open(lock_file, "a") as f:
        # LOCK_EX: Exclusive lock
        # This blocks until the lock is acquired
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            # LOCK_UN: Unlock
            fcntl.flock(f, fcntl.LOCK_UN)
