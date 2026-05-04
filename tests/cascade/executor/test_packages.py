import re

import pytest
from packaging.version import Version

from cascade.executor.runner.packages import (
    PackagesEnv,
    PostverifyIssue,
    ResolutionIssue,
    _is_ignorable_dist,
    _is_ignorable_module,
    _maybe_imported_version,
    _parse_pip_install,
    check_install_result,
    check_run_result,
    run_command,
)
from cascade.low.exceptions import CascadeInternalError, CascadeUserError


def test_run_command() -> None:
    succ_command = ["echo", "you", "shall", "pass"]
    bad1_command = ["you", "shall", "not", "pass"]
    bad2_command = ["uv", "pip", "install", "nonexistentpackagename"]

    run_command(succ_command, check_run_result)

    with pytest.raises(
        CascadeInternalError,
        match=re.escape(
            "command failure: [Errno 2] No such file or directory: 'you' (caused by FileNotFoundError(2, 'No such file or directory'))"
        ),
    ):
        run_command(bad1_command, check_run_result)
    with pytest.raises(
        CascadeUserError,
        match=r"nonexistentpackagename was not found in the package registry",
    ):
        run_command(bad2_command, lambda e: check_install_result(e, True))


def test_parse_pip_install() -> None:
    assumed = "Using Python 3.11.8 environment at: <venv>\nResolved 1 package in 5ms\nUninstalled 1 package in 12ms\nInstalled 1 package in 18ms\n - numpy==2.4.2\n + numpy==2.4.1\n"
    expected = {"numpy": Version("2.4.1")}
    assert _parse_pip_install(assumed) == expected, "failed to parse assumed uv output"
    import numpy

    actual = run_command(["uv", "pip", "install", f"numpy=={numpy.__version__}"], lambda e: check_install_result(e, True)).stderr
    expected = {}
    assert _parse_pip_install(actual) == expected, "failed to parse actual uv output"
    # TODO it would be nice to actually do some pip install here, but im reluctant to do that in a unit test. Some for test_postinstall_verify


def test_maybe_imported_version() -> None:
    import numpy

    assert _maybe_imported_version("numpy") == Version(numpy.__version__)
    import cascade

    assert _maybe_imported_version("cascade") == None  # whoopsie, we dont declare __version__ on cascade
    import earthkit.workflows

    if not hasattr(earthkit.workflows, "__version__"):
        assert False  # just to satisfy ty
    assert _maybe_imported_version("earthkit.workflows") == Version(earthkit.workflows.__version__)


def test_postinstall_verify() -> None:
    import numpy

    env = PackagesEnv()
    goodie = "Using Python 3.11.8 environment at: <venv>\nResolved 1 package in 5ms\nUninstalled 1 package in 12ms\nInstalled 1 package in 18ms\n - numpy==2.4.2\n + numpy=={numpy.__version__}\n"
    issues = env._postinstall_verify(goodie)
    assert not issues
    baddie = "Using Python 3.11.8 environment at: <venv>\nResolved 1 package in 5ms\nUninstalled 1 package in 12ms\nInstalled 1 package in 18ms\n - numpy==2.4.2\n + numpy==1.0.0\n"
    issues = env._postinstall_verify(baddie)
    assert issues == [
        PostverifyIssue(dist_name="numpy", desired_version=Version("1.0.0"), mod_issues=[("numpy", Version(numpy.__version__))])
    ]


def test_prefer_installed() -> None:
    import numpy
    import packaging
    import pytest

    env = PackagesEnv()
    assert set(env._prefer_installed(["numpy"])) >= {
        f"numpy=={numpy.__version__}",
        f"packaging=={packaging.__version__}",
        f"pytest=={pytest.__version__}",
    }, "unrestricted spec of installed pkg didnt resolve to installed pkg"
    assert set(env._prefer_installed(["numpy==1.0.0"])) >= {"numpy==1.0.0"}, (
        "exact pin spec of installed pkg which is compatible didnt match installed version"
    )
    assert set(env._prefer_installed([f"numpy>={numpy.__version__}"])) >= {f"numpy=={numpy.__version__}"}, (
        "range spec of installed pkg which is compatible didnt match installed version"
    )
    assert set(env._prefer_installed(["grumpy==42.0.0"])) >= {"grumpy==42.0.0"}, "spec of uninstalled package got dropped"


def test_check_conflicts() -> None:
    import numpy

    env = PackagesEnv()

    # No conflict: single package
    assert env._check_conflicts([f"numpy=={numpy.__version__}"]) == []

    # No conflict: two compatible specs for the same package
    assert env._check_conflicts([f"numpy=={numpy.__version__}", f"numpy>={numpy.__version__}"]) == []

    # Conflict: two different exact pins
    issues = env._check_conflicts(["numpy==1.0.0", "numpy==2.0.0"])
    assert len(issues) == 1
    assert isinstance(issues[0], ResolutionIssue)
    assert "numpy" in issues[0].because

    # Conflict: import pin vs incompatible user request
    issues = env._check_conflicts([f"numpy=={numpy.__version__}", "numpy==1.0.0"])
    assert len(issues) == 1
    assert "numpy" in issues[0].because

    # No conflict: different packages
    assert env._check_conflicts(["numpy==1.0.0", "packaging==21.0"]) == []

    # No conflict: unconstrained spec does not trigger
    assert env._check_conflicts(["numpy", f"numpy=={numpy.__version__}"]) == []


def test_is_ignorable_module() -> None:
    assert _is_ignorable_module("os")
    assert _is_ignorable_module("sys")
    assert _is_ignorable_module("_pytest")
    assert _is_ignorable_module("__main__")
    assert not _is_ignorable_module("numpy")
    assert not _is_ignorable_module("packaging")


def test_is_ignorable_dist() -> None:
    # A normal published package should not be ignorable
    assert not _is_ignorable_dist("numpy", Version("2.4.1"))
    # dev/local versions (like our own editable installs) should be ignorable
    assert _is_ignorable_dist("earthkit-workflows", Version("0.0.0.dev0"))
    assert _is_ignorable_dist("something", Version("1.0.0+local"))
    assert _is_ignorable_dist("something", Version("0.0.0"))


def test_import_pins_includes_imported_modules() -> None:
    import numpy
    import packaging

    env = PackagesEnv()
    pins = env._import_pins()

    assert "numpy" in pins
    assert pins["numpy"] == Version(str(pins["numpy"]))  # is a valid Version
    assert "packaging" in pins

    # Our own editable install should be excluded (dev/local version)
    assert "earthkit-workflows" not in pins


def test_dist_name_cache_is_populated() -> None:
    import numpy

    env = PackagesEnv()
    # Before any call, cache is empty
    assert "numpy" not in env._dist_name_cache

    env._import_pins()

    # After _import_pins, numpy should be cached
    assert "numpy" in env._dist_name_cache
    assert env._dist_name_cache["numpy"] == "numpy"
