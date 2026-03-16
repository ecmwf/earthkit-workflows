import re

import pytest
from packaging.version import Version

from cascade.executor.runner.packages import (
    InstallIssue,
    _get_dist_modules,
    _maybe_module_version,
    _parse_pip_install,
    _postinstall_verify,
    _prefer_installed,
    run_command,
)
from cascade.low.exceptions import CascadeInternalError, CascadeUserError


def test_run_command() -> None:
    succ_command = ["echo", "you", "shall", "pass"]
    bad1_command = ["you", "shall", "not", "pass"]
    bad2_command = ["uv", "pip", "install", "nonexistentpackagename"]

    run_command(succ_command)

    with pytest.raises(
        CascadeInternalError,
        match=re.escape(
            "command failure: [Errno 2] No such file or directory: 'you'"
            " (caused by FileNotFoundError(2, 'No such file or directory'))"
        ),
    ):
        run_command(bad1_command)
    with pytest.raises(
        CascadeUserError,
        match=r"nonexistentpackagename was not found in the package registry",
    ):
        run_command(bad2_command)


def test_parse_pip_install() -> None:
    assumed = (
        "Using Python 3.11.8 environment at: <venv>\nResolved 1 package in 5ms\n"
        "Uninstalled 1 package in 12ms\nInstalled 1 package in 18ms\n - numpy==2.4.2\n + numpy==2.4.1\n"
    )
    expected = {"numpy": Version("2.4.1")}
    assert _parse_pip_install(assumed) == expected, "failed to parse assumed uv output"
    import numpy

    _, actual = run_command(["uv", "pip", "install", f"numpy=={numpy.__version__}"])
    expected = {}
    assert _parse_pip_install(actual) == expected, "failed to parse actual uv output"
    # TODO it would be nice to actually do some pip install here, but im reluctant to do that in a unit test.
    # Same for test_postinstall_verify


def test_get_dist_modules() -> None:
    assert _get_dist_modules("numpy") == ["numpy"]
    assert sorted(_get_dist_modules("earthkit-workflows")) == sorted(["cascade", "earthkit"])


def test_maybe_module_version() -> None:
    import numpy

    assert _maybe_module_version("numpy") == Version(numpy.__version__)
    assert _maybe_module_version("cascade") is None  # whoopsie, we dont declare __version__ on cascade
    import earthkit.workflows

    if not hasattr(earthkit.workflows, "__version__"):
        assert False  # just to satisfy ty
    assert _maybe_module_version("earthkit.workflows") == Version(earthkit.workflows.__version__)


def test_postinstall_verify() -> None:
    import numpy

    goodie = (
        "Using Python 3.11.8 environment at: <venv>\nResolved 1 package in 5ms\n"
        f"Uninstalled 1 package in 12ms\nInstalled 1 package in 18ms\n - numpy==2.4.2\n + numpy=={numpy.__version__}\n"
    )
    issues = _postinstall_verify(goodie)
    assert not issues
    baddie = (
        "Using Python 3.11.8 environment at: <venv>\nResolved 1 package in 5ms\n"
        "Uninstalled 1 package in 12ms\nInstalled 1 package in 18ms\n - numpy==2.4.2\n + numpy==1.0.0\n"
    )
    issues = _postinstall_verify(baddie)
    assert issues == [
        InstallIssue(
            dist_name="numpy", desired_version=Version("1.0.0"), mod_issues=[("numpy", Version(numpy.__version__))]
        )
    ]


def test_prefer_installed() -> None:
    import numpy

    assert list(_prefer_installed(["numpy"])) == [f"numpy=={numpy.__version__}"]
    assert list(_prefer_installed(["numpy==1.0.0"])) == ["numpy==1.0.0"]
    assert list(_prefer_installed([f"numpy>={numpy.__version__}"])) == [f"numpy=={numpy.__version__}"]
