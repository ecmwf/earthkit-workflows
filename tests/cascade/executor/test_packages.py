import re

import pytest

from cascade.executor.runner.packages import run_command


def test_run_command() -> None:
    succ_command = ["echo", "you", "shall", "pass"]
    bad1_command = ["you", "shall", "not", "pass"]
    bad2_command = ["uv", "pip", "install", "nonexistentpackagename"]

    run_command(succ_command)

    with pytest.raises(
        ValueError,
        match=re.escape("command failure: [Errno 2] No such file or directory: 'you'"),
    ):
        run_command(bad1_command)
    with pytest.raises(
        ValueError,
        match=r"nonexistentpackagename was not found in the package registry",
    ):
        run_command(bad2_command)
