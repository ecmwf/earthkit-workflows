import re
from unittest.mock import patch

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
    _pip_index_flags,
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

    env = PackagesEnv({})
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

    env = PackagesEnv({})
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

    env = PackagesEnv({})

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

    env = PackagesEnv({})
    pins = env._import_pins()

    assert "numpy" in pins
    assert pins["numpy"] == Version(str(pins["numpy"]))  # is a valid Version
    assert "packaging" in pins

    # Our own editable install should be excluded (dev/local version)
    assert "earthkit-workflows" not in pins


def test_dist_name_cache_is_populated() -> None:
    import numpy

    env = PackagesEnv({})
    # Before any call, cache is empty
    assert "numpy" not in env._dist_name_cache

    env._import_pins()

    # After _import_pins, numpy should be cached
    assert "numpy" in env._dist_name_cache
    assert env._dist_name_cache["numpy"] == "numpy"


# ---------------------------------------------------------------------------
# Tests for the _is_already_satisfied fast-path and the extend() skip logic
# ---------------------------------------------------------------------------


def _make_env_with_cache(*specs: tuple[str, str]) -> PackagesEnv:
    """Create a PackagesEnv whose _installed cache is pre-populated with (name, version) pairs."""
    env = PackagesEnv({})
    for name, ver in specs:
        env._installed[name] = Version(ver)
    return env


def test_is_already_satisfied_bare_name_in_cache() -> None:
    env = _make_env_with_cache(("mylib", "1.2.3"))
    # Bare name with no constraint - should be satisfied by anything installed
    assert env._is_already_satisfied(["mylib"])


def test_is_already_satisfied_exact_pin_matches() -> None:
    env = _make_env_with_cache(("mylib", "1.2.3"))
    assert env._is_already_satisfied(["mylib==1.2.3"])


def test_is_already_satisfied_exact_pin_mismatches() -> None:
    env = _make_env_with_cache(("mylib", "1.2.3"))
    assert not env._is_already_satisfied(["mylib==1.0.0"])


def test_is_already_satisfied_range_satisfied() -> None:
    env = _make_env_with_cache(("mylib", "1.2.3"))
    assert env._is_already_satisfied(["mylib>=1.0.0"])
    assert env._is_already_satisfied(["mylib>=1.0.0,<2.0.0"])


def test_is_already_satisfied_range_not_satisfied() -> None:
    env = _make_env_with_cache(("mylib", "1.2.3"))
    assert not env._is_already_satisfied(["mylib>=2.0.0"])
    assert not env._is_already_satisfied(["mylib<1.0.0"])


def test_is_already_satisfied_multiple_all_satisfied() -> None:
    env = _make_env_with_cache(("aaa", "1.0.0"), ("bbb", "2.5.0"))
    assert env._is_already_satisfied(["aaa==1.0.0", "bbb>=2.0"])


def test_is_already_satisfied_multiple_one_fails() -> None:
    env = _make_env_with_cache(("aaa", "1.0.0"), ("bbb", "2.5.0"))
    # bbb is satisfied but aaa is not
    assert not env._is_already_satisfied(["aaa==9.9.9", "bbb>=2.0"])


def test_is_already_satisfied_not_installed() -> None:
    env = PackagesEnv({})
    # A package that almost certainly does not exist in this venv
    assert not env._is_already_satisfied(["totally-nonexistent-xyz-package"])


def test_is_already_satisfied_not_in_cache_returns_false() -> None:
    """Packages not in _installed are not satisfied — no importlib fallback."""
    import numpy

    env = PackagesEnv({})
    # numpy is NOT in env._installed (we did not add it manually)
    assert "numpy" not in env._installed
    # Without the importlib fallback, _is_already_satisfied returns False for anything not cached
    assert not env._is_already_satisfied([f"numpy=={numpy.__version__}"])
    assert not env._is_already_satisfied(["numpy"])


def test_extend_skips_pip_when_already_satisfied() -> None:
    """When every requested spec is already installed, extend() must not invoke pip."""
    import numpy

    env = _make_env_with_cache(("numpy", numpy.__version__))
    with patch("cascade.executor.runner.packages.run_command") as mock_run:
        env.extend([f"numpy=={numpy.__version__}"])
        mock_run.assert_not_called()


def test_extend_calls_pip_when_not_satisfied() -> None:
    """When a spec is NOT satisfied, extend() must still invoke pip (and may fail)."""
    env = PackagesEnv({})
    with patch("cascade.executor.runner.packages.run_command") as mock_run:
        # Simulate pip returning empty stderr (no-op install) so extend() completes cleanly
        mock_run.return_value = type("R", (), {"stderr": "", "stdout": "", "returncode": 0})()
        env.extend(["totally-nonexistent-xyz==999.0.0"])
        mock_run.assert_called_once()


def test_extend_backfills_installed_after_noop_pip() -> None:
    """After a no-op pip run, the requested package should be added to _installed."""
    import numpy

    env = PackagesEnv({})
    assert "numpy" not in env._installed

    with patch("cascade.executor.runner.packages.run_command") as mock_run:
        mock_run.return_value = type("R", (), {"stderr": "", "stdout": "", "returncode": 0})()
        env.extend([f"numpy=={numpy.__version__}"])

    # Back-fill: numpy should now be in _installed even though pip reported no changes
    assert "numpy" in env._installed
    assert env._installed["numpy"] == Version(numpy.__version__)

    # Subsequent call should skip pip entirely
    with patch("cascade.executor.runner.packages.run_command") as mock_run2:
        env.extend([f"numpy=={numpy.__version__}"])
        mock_run2.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _pip_index_flags and pip_indices propagation
# ---------------------------------------------------------------------------


def test_pip_index_flags_empty() -> None:
    assert _pip_index_flags([]) == []


def test_pip_index_flags_local_abs_path(tmp_path) -> None:
    flags = _pip_index_flags([str(tmp_path)])
    assert flags == ["--find-links", str(tmp_path)]


def test_pip_index_flags_url() -> None:
    url = "https://test.pypi.org/simple/"
    flags = _pip_index_flags([url])
    assert flags == ["--extra-index-url", url]


def test_pip_index_flags_mixed(tmp_path) -> None:
    local = str(tmp_path)
    url = "https://test.pypi.org/simple/"
    flags = _pip_index_flags([local, url])
    assert flags == ["--find-links", local, "--extra-index-url", url]


def test_pip_index_flags_relative_path_treated_as_url() -> None:
    # Relative paths are not absolute, so treated as index URLs
    flags = _pip_index_flags(["relative/path"])
    assert flags == ["--extra-index-url", "relative/path"]


def test_extend_includes_index_flags_in_pip_call(tmp_path) -> None:
    """When pip_indices are set, extend() must include them in the pip command."""
    env = PackagesEnv({}, pip_indices=[str(tmp_path), "https://test.pypi.org/simple/"])
    with patch("cascade.executor.runner.packages.run_command") as mock_run:
        mock_run.return_value = type("R", (), {"stderr": "", "stdout": "", "returncode": 0})()
        env.extend(["totally-nonexistent-xyz==999.0.0"])
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "--find-links" in call_args
        assert str(tmp_path) in call_args
        assert "--extra-index-url" in call_args
        assert "https://test.pypi.org/simple/" in call_args


def test_extend_no_index_flags_when_empty() -> None:
    """When pip_indices is empty, no index flags appear in the pip command."""
    env = PackagesEnv({})
    with patch("cascade.executor.runner.packages.run_command") as mock_run:
        mock_run.return_value = type("R", (), {"stderr": "", "stdout": "", "returncode": 0})()
        env.extend(["totally-nonexistent-xyz==999.0.0"])
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "--find-links" not in call_args
        assert "--extra-index-url" not in call_args
